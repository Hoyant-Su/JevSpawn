import argparse
from collections import defaultdict
from copy import copy
import json
import math
import os
from pathlib import Path
import time
from types import SimpleNamespace

import torch
import torch.distributed as dist

from baselines.common.config import SharedConfig
from baselines.common.parallel_run import initialize_parallel
from jev_spawn.algo.structured import common_prefix
from jev_spawn.infra.cached_suffix import ragged_suffix
from jev_spawn.infra.finite_batch import eager_finite_tail, score_finite_with_tail
from jev_spawn.infra.readout_labels import AdmittedPrompt
from jev_spawn.runtime.prefix_cache import PrefixCache
from jev_spawn.runtime.state import event_history, history_frontier
from jev_spawn.schema import CONTROLLER, controller_prefix, controller_prompts


def prepare(settings, backend, shared):
    rows = json.loads(Path(settings['requests']).read_text())
    serialization = json.loads(Path(settings['method']).read_text())['settings']['rollout']['execution']['serialization']
    assert len(rows) == settings['expected_batch_size'] == shared.runtime.batch_size
    assert len({row['task_id'] for row in rows}) == len(rows)
    cohorts = [[] for _ in settings['expected_turns']]
    for row in rows:
        source = Path(row['source'])
        original_task = json.loads(source.read_text())
        assert original_task['task_id'] == row['task_id']
        original_rounds = original_task['trace']['rounds']
        original_events = {event['id']: event for turn in original_rounds
            for child in turn.get('children', {}).values() for step in child
            if 'observations' in step for event in step['observations']}
        assert [record['turn'] for record in row['turns']] == settings['expected_turns']
        previous_events, previous_checkpoint = [], ()
        for index, record in enumerate(row['turns']):
            recovered, field = record['recovered_state'], record['request']
            original = next(turn['frontier_request'] for turn in original_rounds
                            if turn['turn'] == record['turn'])
            assert original == record['original_request']
            events = recovered['execution_events']
            assert events == [original_events[identity]
                              for identity in json.loads(original['state'])['execution_order']]
            assert events[:len(previous_events)] == previous_events
            assert len(events) > len(previous_events)
            state, branches = history_frontier(recovered)
            assert field['history'] == event_history(events)
            assert field['state'] == json.dumps(state, **serialization)
            assert field['options'] == [{'id': option['id'], 'description': json.dumps(
                branches[option['id']], **serialization)} for option in record['original_request']['options']]
            prompt = controller_prompts([field['state']], field['question'], field['options'],
                list(backend.answer_labels[:len(field['options'])]), CONTROLLER['output_instruction'],
                contexts=[field['context']], histories=[field['history']])[0]
            rendered = backend._render([prompt], CONTROLLER['system'])[0]
            tokens = backend.tokenizer(rendered, add_special_tokens=False)['input_ids']
            prefixes = backend._render([controller_prefix(field['context'], history)
                for history in ('', field['history'])], CONTROLLER['system'])
            prefix_tokens = backend.tokenizer(prefixes, add_special_tokens=False)['input_ids']
            root_length, checkpoint_length = [common_prefix([tokens, prefix]) for prefix in prefix_tokens]
            assert 0 < root_length < checkpoint_length < len(tokens) <= shared.model.max_input_tokens
            checkpoint = tuple(tokens[:checkpoint_length])
            assert checkpoint[:len(previous_checkpoint)] == previous_checkpoint
            cohorts[index].append(SimpleNamespace(task_id=row['task_id'], field=field,
                admitted=AdmittedPrompt(rendered, tuple(tokens)), root_tokens=tuple(tokens[:root_length]),
                checkpoint_tokens=checkpoint))
            previous_events, previous_checkpoint = events, checkpoint
    return cohorts


def compare(reference, candidate, settings):
    rows = []
    for original, cached in zip(reference['groups'][0], candidate['groups'][0], strict=True):
        assert original['id'] == cached['id'] and original['option_ids'] == cached['option_ids']
        error = max(abs(a - b) for a, b in zip(original['option_logits'], cached['option_logits'], strict=True))
        finite = all(math.isfinite(value) for value in original['option_logits'] + cached['option_logits'])
        rows.append({'id': original['id'], 'max_logit_error': error,
                     'reference_choice': original['choice'], 'candidate_choice': cached['choice'],
                     'reference_logits': original['option_logits'], 'candidate_logits': cached['option_logits'],
                     'choice_agreement': original['choice'] == cached['choice'],
                     'finite_logits': finite,
                     'logit_pass': finite and error <= settings['logit_max_absolute_error']})
    agreement = sum(row['choice_agreement'] for row in rows) / len(rows)
    logit_pass = all(row['logit_pass'] for row in rows)
    choice_pass = agreement >= settings['required_choice_agreement']
    return {'rows': rows, 'max_logit_error': max(row['max_logit_error'] for row in rows),
            'choice_agreement': agreement, 'logit_tolerance': settings['logit_max_absolute_error'],
            'required_choice_agreement': settings['required_choice_agreement'],
            'logit_pass': logit_pass, 'choice_pass': choice_pass, 'passed': logit_pass and choice_pass}


def base_lengths(requests):
    groups = defaultdict(list)
    for request in requests:
        groups[request.task_id, request.field['context']].append(request)
    lengths = {key: common_prefix([request.root_tokens for request in group])
               for key, group in groups.items()}
    return [lengths[request.task_id, request.field['context']] for request in requests]


def score(backend, requests, base, history):
    torch.cuda.synchronize(backend.device)
    torch.cuda.reset_peak_memory_stats(backend.device)
    started = time.perf_counter()
    result = score_finite_with_tail(backend, requests, base_lengths(requests),
        base, history, eager_finite_tail, ragged_suffix, common_prefix,
        physical_batch_size=len(requests))
    torch.cuda.synchronize(backend.device)
    elapsed = time.perf_counter() - started
    rankmax = torch.tensor(elapsed, device=backend.device, dtype=torch.float64)
    dist.all_reduce(rankmax, op=dist.ReduceOp.MAX)
    result.pop('device_probabilities')
    result.update(rankmax_seconds=rankmax.item(), task_ids=[r.task_id for r in requests],
                  requested_checkpoint_tokens=[len(r.checkpoint_tokens) for r in requests])
    assert result['batch_size'] == result['physical_batch_size'] == len(requests)
    return result


def replay(backend, cohorts, variant, settings, progress, phase, audit_immutable=False):
    base = PrefixCache(settings['root_cache_capacity'])
    history = PrefixCache(settings['history_cache_capacity'])
    records, originals = [], []
    for index, cohort in enumerate(cohorts):
        requests = [copy(request) for request in cohort]
        if variant == 'root':
            history.clear()
            for request in requests:
                request.checkpoint_tokens = request.root_tokens
        result = score(backend, requests, base, history)
        (progress / settings['progress_file'].format(rank=dist.get_rank(), phase=phase,
            variant=variant, turn=settings['expected_turns'][index])).write_text(json.dumps({
                'status': 'computed_not_validated', 'phase': phase, 'variant': variant,
                'turn': settings['expected_turns'][index], 'result': result}, indent=2) + '\n')
        records.append(result)
        if audit_immutable and index == 0:
            originals = [(key, value, value.get_seq_length()) for key, value in history.entries.items()]
    audit, audit_comparison = None, None
    retained_lengths = []
    if audit_immutable:
        assert variant == 'history'
        # Retain the actual earlier hybrid states even when normal capacity evicts their keys.
        saved = PrefixCache(settings['history_cache_capacity'])
        for key, value, length in originals:
            current = value.get_seq_length()
            retained_lengths.append({'original_length': length, 'retained_length': current,
                'token_length': len(key[0]), 'passed': current == length == len(key[0])})
            saved.entries[key] = value
        audit = score(backend, cohorts[0], base, saved)
        (progress / settings['progress_file'].format(rank=dist.get_rank(), phase=phase,
            variant='retained_checkpoint', turn=settings['expected_turns'][0])).write_text(json.dumps({
                'status': 'computed_not_validated', 'phase': phase,
                'variant': 'retained_checkpoint', 'result': audit}, indent=2) + '\n')
        audit_comparison = compare(records[0], audit, settings)
        saved.clear()
    checks = {'history_reused': variant != 'history' or records[1]['reused_state_tokens'] > 0}
    if audit_immutable:
        checks.update(retained_lengths_unchanged=bool(retained_lengths) and
                      all(row['passed'] for row in retained_lengths),
                      retained_checkpoint_hits=all(audit['persistent_prefix_hits']),
                      immutable_logits_pass=audit_comparison['passed'])
    base.clear()
    history.clear()
    torch.cuda.synchronize(backend.device)
    return {'records': records, 'immutable_checkpoint_logits_audit': audit,
            'immutable_checkpoint_comparison': audit_comparison,
            'retained_checkpoint_lengths': retained_lengths,
            'checks': checks, 'passed': all(checks.values()),
            'timed_turn_seconds': sum(record['rankmax_seconds'] for record in records)}


@torch.inference_mode()
def main(settings):
    shared = SharedConfig.load(settings['shared_config'])
    assert shared.runtime.world_size == settings['expected_world_size']
    assert shared.runtime.root_batch_size == settings['root_cache_capacity'] == settings['history_cache_capacity']
    backend, commands, startup = initialize_parallel(shared,
        json.loads(Path(settings['parallel_settings']).read_text()))
    output = Path(settings['output'])
    progress = output / settings['progress_directory'].format(run_id=os.environ['TORCHELASTIC_RUN_ID'])
    progress.mkdir(parents=True, exist_ok=True)
    cohorts = prepare(settings, backend, shared)
    phases = []
    for phase in settings['phases']:
        variants = {name: replay(backend, cohorts, name, settings, progress, phase['name'])
                    for name in phase['order']}
        comparisons = []
        for turn, original, candidate in zip(settings['expected_turns'], variants['root']['records'],
                                            variants['history']['records'], strict=True):
            comparison = compare(original, candidate, settings)
            comparisons.append({'turn': turn, **comparison,
                'root_computed_tokens': original['computed_input_tokens'],
                'history_computed_tokens': candidate['computed_input_tokens'],
                'root_rankmax_seconds': original['rankmax_seconds'],
                'history_rankmax_seconds': candidate['rankmax_seconds']})
        reduced = comparisons[-1]['history_computed_tokens'] < comparisons[-1]['root_computed_tokens']
        phases.append({**phase, 'variants': variants, 'comparisons': comparisons,
            'later_turn_computed_tokens_reduced': reduced,
            'passed': reduced and all(row['passed'] for row in comparisons)
                      and all(variant['passed'] for variant in variants.values())})
        if commands.is_leader:
            print(json.dumps({'phase': phase['name'], 'measured': phase['measured'],
                              'comparisons': comparisons}), flush=True)
    immutable_audit = replay(backend, cohorts, 'history', settings, progress,
                            'immutable_audit', audit_immutable=True)
    choices = [[[row['choice'] for row in result['groups'][0]] for result in
                phase['variants']['history']['records']] for phase in phases]
    ranks = [None] * shared.runtime.world_size
    local_passed = all(phase['passed'] for phase in phases) and immutable_audit['passed']
    dist.all_gather_object(ranks, {'choices': choices, 'passed': local_passed}, group=commands.control_group)
    rank_choices_equal = all(rank['choices'] == choices for rank in ranks)
    passed = rank_choices_equal and all(rank['passed'] for rank in ranks)
    report = {'status': 'passed' if passed else 'failed', 'passed': passed,
        'settings': settings, 'startup': startup, 'phases': phases,
        'rank_choices_equal': rank_choices_equal, 'rank_pass_flags': [rank['passed'] for rank in ranks],
        'immutable_checkpoint_audit': immutable_audit,
        'task_ids': [request.task_id for request in cohorts[0]],
        'scope': 'Paired same-instance 27B finite-request replay, not task accuracy or inter-token latency.',
        'timing_scope': 'Synchronized per-turn compute with rank maximum; loading, preparation and immutable-state audit excluded. Warmup reported separately.',
        'unique_real_requests': len(cohorts) * len(cohorts[0])}
    (output / settings['rank_file'].format(rank=dist.get_rank())).write_text(json.dumps(report, indent=2) + '\n')
    if commands.is_leader:
        (output / settings['summary_file']).write_text(json.dumps(report, indent=2) + '\n')
    dist.destroy_process_group(commands.control_group)
    dist.destroy_process_group()
    assert passed, f"Replay qualification failed; full diagnostic report written to {output}"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    main(json.loads(parser.parse_args().config.read_text()))
