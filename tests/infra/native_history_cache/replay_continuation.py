import argparse
from copy import deepcopy
import json
from pathlib import Path
import time

import torch
import torch.distributed as dist

from baselines.common.config import SharedConfig
from baselines.common.parallel_run import initialize_parallel
from jev_spawn.infra.prompts import load_prompt
from jev_spawn.infra.readout_labels import AdmittedPrompt
from jev_spawn.infra.stable_finite_graph import StableFiniteGraphTail
from jev_spawn.runtime.prefix_cache import PrefixCache
from tests.infra.native_history_cache.continuation import ContinuationTail, continue_messages
from tests.infra.native_history_cache.replay import requests_from_record
from tests.infra.native_suffix_graph.qualify import compare, measure, tensors


def make_sequence(backend, settings, sources, tail, roots):
    protocol = settings['continuation']
    prompts = load_prompt(protocol['prompts'])
    anchors, sequence = {}, []
    for source in sources:
        requests, lengths = requests_from_record(backend, source)
        for request in requests:
            field = request.field
            parent = protocol['parents'].get(field['id'], protocol['field_parent'])
            anchor = anchors.get((request.task_id, parent))
            compatible = (anchor is not None and anchor['context'] == field['context']
                          and anchor['history'] == field['history'])
            if field['id'] != protocol['reset_field'] and compatible:
                request.messages = continue_messages(anchor, field, backend.answer_labels, prompts)
                rendered = backend.tokenizer.apply_chat_template(request.messages, tokenize=False,
                    add_generation_prompt=True, enable_thinking=False)
                tokens = backend.tokenizer(rendered, add_special_tokens=False)['input_ids']
                assert len(tokens) <= backend.config['max_input_tokens']
                assert tuple(tokens[:len(anchor['tokens'])]) == anchor['tokens']
                request.admitted = AdmittedPrompt(rendered, tuple(tokens))
                field['continuation_parent'] = parent
        result = tail.score(requests, lengths, roots)
        answers = [answer for group in result['groups'] for answer in group]
        for request, answer in zip(requests, answers, strict=True):
            field = request.field
            if field['id'] in protocol['anchors']:
                selected = answer['option_ids'].index(answer['choice'])
                anchors[(request.task_id, field['id'])] = {
                    'context': field['context'], 'history': field['history'],
                    'messages': request.messages, 'label': backend.answer_labels[selected],
                    'tokens': request.admitted.tokens}
        sequence.append((requests, lengths))
    return sequence


@torch.inference_mode()
def run(settings):
    shared = SharedConfig.load(settings['shared_config'])
    backend, commands, startup = initialize_parallel(
        shared, json.loads(Path(settings['parallel_settings']).read_text()))
    service = json.loads(Path(settings['service']).read_text())['settings']
    output = Path(settings['output'])
    output.mkdir(parents=True, exist_ok=True)
    report = {'settings': settings, 'startup': startup, 'sequences': []}
    for sources in settings['sequences']:
        roots = PrefixCache(shared.runtime.root_batch_size)
        reference = StableFiniteGraphTail(backend, shared.runtime,
            PrefixCache(shared.runtime.root_batch_size), service['state_copy'], service['graph_shape'])
        candidate = ContinuationTail(backend, shared.runtime,
            PrefixCache(shared.runtime.root_batch_size), service['state_copy'], service['graph_shape'],
            continuation=settings['continuation'])
        tails = dict(zip(settings['arms'], (reference, candidate), strict=True))
        sequence = make_sequence(backend, settings, sources, reference, roots)
        diagnostics = []

        def reset():
            for tail in tails.values():
                tail.prefix_cache.clear()
            candidate.anchor_states.clear()

        reset()
        for source, (requests, lengths) in zip(sources, sequence, strict=True):
            values, logits = {}, {}
            for name, tail in tails.items():
                started = time.perf_counter()
                result = tail.score(requests, lengths, roots)
                torch.cuda.synchronize(backend.device)
                values[name] = {key: result[key] for key in settings['workload_fields']}
                values[name]['first_wall_seconds'] = time.perf_counter() - started
                values[name]['choices'] = [answer['choice'] for group in result['groups'] for answer in group]
                logits[name] = tail.last_logits.clone()
            keys = tuple(reference.prefix_cache.entries)
            assert keys == tuple(candidate.prefix_cache.entries)
            diagnostics.append({'source': source, 'task_ids': [r.task_id for r in requests],
                'requests': [{'task_id': r.task_id, 'field': deepcopy(r.field), 'messages': r.messages,
                              'input_ids': list(r.admitted.tokens)} for r in requests],
                'workload': values, 'logits': compare(list(logits.values())[:1], list(logits.values())[1:]),
                'choices_equal': values[settings['arms'][0]]['choices'] == values[settings['arms'][1]]['choices'],
                'native_states': compare(
                    [t for key in keys for t in tensors(reference.prefix_cache.entries[key])],
                    [t for key in keys for t in tensors(candidate.prefix_cache.entries[key])])})
        timings = []
        for _ in range(settings['repetitions']):
            reset()
            timings.append([{name: measure(lambda: tail.score(requests, lengths, roots),
                backend.device, settings['measurements_per_cohort']) for name, tail in tails.items()}
                for requests, lengths in sequence])
        report['sequences'].append({'sources': sources, 'diagnostics': diagnostics, 'timings': timings})
        (output / settings['rank_file'].format(rank=dist.get_rank())).write_text(json.dumps(report, indent=2) + '\n')
        for tail in tails.values():
            tail.graphs.clear()
    dist.destroy_process_group(commands.control_group)
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    run(json.loads(parser.parse_args().config.read_text()))
