import argparse
from collections import OrderedDict
import json
from pathlib import Path

from transformers import AutoTokenizer
import yaml

from baselines.common.context_window import truncate_prompt
from jev_spawn.algo.structured import common_prefix
from jev_spawn.infra.configuration import CORE
from jev_spawn.infra.prompts import load_prompt
from jev_spawn.schema import CONTROLLER, controller_prompts, controller_prefix


def pairs(value):
    if isinstance(value, dict):
        for name in ('frontier', 'operation'):
            if name + '_decision' in value:
                yield value[name + '_request'], value[name + '_decision']
        for key in ('decisions', 'outputs'):
            if 'requests' in value and key in value:
                yield from zip(value['requests'], value[key], strict=True)
        for child in value.values():
            yield from pairs(child)
    elif isinstance(value, list):
        for child in value:
            yield from pairs(child)


def collect(directory, settings, shared, tokenizer):
    runtime = json.loads((directory / 'session-0000/runtime.json').read_text())['service']
    protocol = json.loads((directory / 'protocol.json').read_text())
    CONTROLLER['option_template'] = protocol['prompts']['option_template']
    system = CONTROLLER['system'] + '\n\n' + load_prompt('shared.runtime_contract').format(
        contract=json.dumps(runtime['runtime_contract']))
    inference = json.loads(Path(settings['inference_config']).read_text())
    window = inference['settings']['input_window']
    unique = {}
    for path in sorted(directory.glob(settings['task_glob'])):
        task = json.loads(path.read_text())
        for field, decision in pairs(task['trace']):
            if 'choice' in decision and decision['input_tokens']:
                key = (task['task_id'], decision['id'], decision['submitted_monotonic'], decision['delivered_monotonic'])
                unique[key] = (field, decision)
    batches = sorted((batch for path in sorted(directory.glob('session-*/batches.json'))
        for batch in json.loads(path.read_text()) if batch.get('operation') == 'finite'),
        key=lambda batch: batch['started_monotonic'])
    cohorts = [[] for _ in batches]
    roots = {}
    truncations = 0
    for (task_id, identity, submitted, delivered), (field, decision) in sorted(unique.items(), key=lambda pair: pair[0][2]):
        prompt, = controller_prompts([field['state']], field['question'], field['options'],
            runtime['candidate_labels'][:len(field['options'])], CONTROLLER['output_instruction'],
            contexts=[field['context']], histories=[field.get('history', '')])
        messages = [{'role': 'system', 'content': system}, {'role': 'user', 'content': prompt}]
        rendered = tokenizer.apply_chat_template(messages, tokenize=False,
            add_generation_prompt=CORE['backend']['add_generation_prompt'], enable_thinking=CORE['backend']['enable_thinking'])
        tokens = tokenizer(rendered, add_special_tokens=False)['input_ids']
        _, tokens, metadata = truncate_prompt(tokenizer, rendered, tokens, shared['model']['max_input_tokens'],
                                             window, load_prompt(window['prompt']))
        assert len(tokens) == decision['input_tokens'], (task_id, identity, len(tokens), decision['input_tokens'])
        truncations += metadata['omitted_tokens'] > 0
        key = (task_id, field['context'])
        if key not in roots:
            root_text = tokenizer.apply_chat_template([{'role':'system','content':system},
                {'role':'user','content':controller_prefix(field['context'], '')}], tokenize=False,
                add_generation_prompt=CORE['backend']['add_generation_prompt'], enable_thinking=CORE['backend']['enable_thinking'])
            roots[key] = tokenizer(root_text, add_special_tokens=False)['input_ids']
        root_length = common_prefix([tokens, roots[key]])
        matching = [index for index, batch in enumerate(batches) if task_id in batch['task_ids']
                    and submitted <= batch['started_monotonic'] and batch['finished_monotonic'] <= delivered]
        index = max(matching, key=lambda position: batches[position]['finished_monotonic'])
        batch = batches[index]
        task_order = list(dict.fromkeys(batch['task_ids']))
        assert len(task_order) == batch['structured']['root_batch_size']
        observed_roots = dict(zip(task_order, batch['structured']['root_prefix_tokens'], strict=True))
        assert observed_roots[task_id] == root_length - 1
        cohorts[index].append({'task_id':task_id, 'tokens':tuple(tokens), 'root_length':observed_roots[task_id]})
    return cohorts, batches, truncations


def simulate(cohorts, block_size, capacity):
    cache = OrderedDict()
    work, blocks, request_hits, local_hits, peak_entries, retained_prefix_tokens = 0, 0, 0, 0, 0, 0
    for cohort in cohorts:
        needed = OrderedDict()
        touched = OrderedDict()
        residual = 0
        for request in cohort:
            tokens, root = request['tokens'], request['root_length']
            boundaries = range(root + block_size, len(tokens), block_size)
            keys = [(request['task_id'], tokens[:stop]) for stop in boundaries]
            hits = [index for index, key in enumerate(keys) if key in cache]
            start = max(hits) + 1 if hits else 0
            request_hits += bool(hits)
            if hits:
                touched[keys[start - 1]] = None
            for key in keys[start:]:
                local_hits += key in needed
                needed[key] = None
            residual += len(tokens) - (root + len(keys) * block_size)
        work += len(needed) * block_size + residual
        blocks += len(needed)
        for key in [*touched, *needed]:
            cache[key] = None
            cache.move_to_end(key)
            if capacity is not None and len(cache) > capacity:
                cache.popitem(last=False)
        peak_entries = max(peak_entries, len(cache))
        retained_prefix_tokens = max(retained_prefix_tokens, sum(len(key[1]) for key in cache))
    return {'nonroot_computed_tokens':work, 'new_blocks':blocks, 'request_persistent_hits':request_hits,
            'same_cohort_shared_block_occurrences':local_hits, 'peak_entries':peak_entries,
            'peak_sum_full_prefix_tokens':retained_prefix_tokens}


def main(settings):
    shared = yaml.safe_load(Path(settings['shared_config']).read_text())
    assert settings['max_entries'] == shared['runtime']['root_batch_size'] * shared['runtime']['graph_cache_size']
    tokenizer = AutoTokenizer.from_pretrained(shared['model']['path'], **settings['tokenizer'])
    assert settings['include_unlimited_upper_bound']
    block = shared['runtime']['graph_cache_block_tokens']
    rows = []
    for directory in settings['trace_runs']:
        cohorts, batches, truncations = collect(Path(directory), settings, shared, tokenizer)
        off = simulate(cohorts, block, 0)
        bounded = simulate(cohorts, block, settings['max_entries'])
        upper = simulate(cohorts, block, None)
        for variant in (bounded, upper):
            variant['nonroot_token_reduction_vs_fixed_segmentation_off'] = 1 - variant['nonroot_computed_tokens'] / off['nonroot_computed_tokens']
        observed_nonroot = sum(sum(end-root for end, root in zip(batch['structured']['prefix_tokens'],
            batch['structured']['root_prefix_tokens'], strict=True)) + sum(batch['structured']['suffix_tokens']) for batch in batches)
        rows.append({'source':directory, 'traced_requests':sum(map(len,cohorts)),
            'recorded_serviced_rows':sum(batch['batch_size'] for batch in batches),
            'cohort_coverage':[{'traced':len(cohort), 'serviced':batch['batch_size']} for cohort,batch in zip(cohorts,batches,strict=True)],
            'truncated_requests':truncations,'block_tokens':block,'observed_root_only_nonroot_tokens':observed_nonroot,
            'fixed_segmentation_cache_off':off, 'lru32':bounded, 'unlimited_upper_bound':upper})
    result={'settings':settings,'rows':rows,
        'scope':'CPU token-prefix workload simulation, not model execution. Exact task-scoped token prefixes at root-relative fixed block boundaries. Each recorded finite cohort reads the preceding persistent cache snapshot; new same-cohort blocks count once. The residual, including the finite final token, is always computed. Root prefill and its unchanged root cache are excluded from both token denominators.',
        'limitations':['Cache-off and cache-on share the same proposed block segmentation. Neither is a revival of event-boundary stage049; exact GPU numerical qualification is still required.',
            '32 entries is a count bound, not a GPU byte bound. Full-prefix token storage reports the memory implication; recurrent-state bytes are not represented by token counts.',
            'Unlimited reuse is a theoretical upper bound, never a deployment configuration.',
            'Actual legacy root-only execution shares arbitrary cohort common prefixes; the fixed-block-off comparator can differ in workload. Both counts are reported.',
            'Every saved completed decision is included and deduplicated. Coverage against serviced rows is explicit; uncommitted or timed-out requests with no saved field inputs cannot be reconstructed.']}
    Path(settings['output']).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps([{k:v for k,v in row.items() if k!='cohort_coverage'} for row in rows],indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,required=True)
    main(json.loads(parser.parse_args().config.read_text()))
