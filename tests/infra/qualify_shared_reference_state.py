import argparse
from collections import defaultdict
from copy import deepcopy
import json
from pathlib import Path

from transformers import AutoTokenizer
import yaml

from baselines.common.resources import TEMPLATES
from jev_spawn.algo.structured import common_prefix
from jev_spawn.infra.prompts import load_prompt
from jev_spawn.runtime.prefix_cache import PrefixCache
from jev_spawn.runtime.shared_reference_state import relocate, restore
from jev_spawn.runtime.reference_transport import unpack_references
from jev_spawn.schema import CONTROLLER, controller_prompts


def ledger(cohorts, capacity, buckets):
    base, state, final = [PrefixCache(capacity) for _ in range(3)]
    records = []
    for cohort in cohorts:
        groups = defaultdict(list)
        for index, item in enumerate(cohort):
            groups[(item['task_id'], item['field']['context'])].append(index)
        sequences = [item['tokens'] for item in cohort]
        prefixes, bases, owners = [], [], {}
        for owner, indices in enumerate(groups.values()):
            length = common_prefix([sequences[index] for index in indices])
            first = indices[0]
            prefixes.append(sequences[first][:length])
            bases.append(sequences[first][:cohort[first]['base_length']])
            owners.update((index, owner) for index in indices)
        base_by_prefix = {tuple(prefix): root for prefix, root in zip(prefixes, bases, strict=True)}
        work = defaultdict(int)
        hits = defaultdict(list)

        def account(tails):
            nonempty = [tail for tail in tails if tail]
            work['computed_input_tokens'] += sum(map(len, nonempty))
            work['padded_input_tokens'] += max(map(len, nonempty)) * len(nonempty) if nonempty else 0

        def prefill(missing):
            account(missing)
            return list(map(len, missing))

        def extend(missing):
            roots = [base_by_prefix[tuple(prefix)] for prefix in missing]
            lengths, resident = base.get_many(roots, prefill)
            hits['base'].extend(resident)
            account([prefix[length:] for prefix, length in zip(missing, lengths, strict=True)])
            return list(map(len, missing))

        lengths, resident = state.get_many(prefixes, extend)
        hits['state'].extend(resident)
        full = [sequence[:-1] for sequence in sequences]
        source = {tuple(prefix): index for index, prefix in enumerate(full)}

        def complete(missing):
            account([prefix[lengths[owners[source[tuple(prefix)]]]:] for prefix in missing])
            return list(map(len, missing))

        _, resident = final.get_many(full, complete)
        hits['final'].extend(resident)
        work['computed_input_tokens'] += len(sequences)
        work['padded_input_tokens'] += next(size for size in buckets if size >= len(sequences))
        records.append({'logical_input_tokens': sum(map(len, sequences)), **work,
                        'prefix_lengths': list(map(len, prefixes)), 'group_sizes': list(map(len, groups.values())),
                        'hits': dict(hits)})
    return {'totals': {name: sum(row[name] for row in records) for name in
                      ('logical_input_tokens', 'computed_input_tokens', 'padded_input_tokens')}, 'cohorts': records}


def main(config):
    shared = yaml.safe_load(Path(config['shared_config']).read_text())
    transport = json.loads(Path(config['transport']).read_text())
    prompt = load_prompt(config['prompt'])['encoded_state']
    run = Path(config['source_run'])
    protocol = json.loads((run / 'protocol.json').read_text())
    CONTROLLER['option_template'] = protocol['prompts']['option_template']
    runtime = json.loads(next(run.glob('session-*/runtime.json')).read_text())
    labels = runtime['backend']['answer_labels']
    tokenizer = AutoTokenizer.from_pretrained(shared['model']['path'],
        local_files_only=config['tokenizer']['local_files_only'], padding_side=config['tokenizer']['padding_side'])
    originals, changed, lengths, failures = [], [], [], []
    for path in sorted(run.glob(config['input_glob'])):
        cohort = json.loads(path.read_text())['requests']
        transformed = []
        for item in cohort:
            replacement = deepcopy(item)
            field = replacement['field']
            segments = relocate(field['input'], transport, config['shared_keys'])
            original = unpack_references(field['input'], transport)
            decoded = restore(segments, transport)
            assert json.dumps(original, **config['canonical_serialization']) == json.dumps(decoded, **config['canonical_serialization'])
            state = prompt.format(**{key: json.dumps(value, **config['serialization']) for key, value in segments.items()})
            field['state'] = TEMPLATES[config['state_template_key']].format(context=field['context'], state=state)
            user, = controller_prompts([field['state']], field['question'], field['options'],
                labels[:len(field['options'])], CONTROLLER['output_instruction'])
            replacement['messages'] = [{'role': 'system', 'content': CONTROLLER['system']}, {'role': 'user', 'content': user}]
            replacement['rendered'] = tokenizer.apply_chat_template(replacement['messages'], tokenize=False,
                add_generation_prompt=config['tokenizer']['add_generation_prompt'], enable_thinking=shared['generation']['enable_thinking'])
            replacement['tokens'] = tokenizer(replacement['rendered'], add_special_tokens=config['tokenizer']['add_special_tokens'])['input_ids']
            assert replacement['tokens'][:item['base_length']] == item['tokens'][:item['base_length']]
            assert field['options'] == item['field']['options'] and field['question'] == item['field']['question']
            replacement['segments'] = segments
            replacement['input_tokens'] = len(replacement['tokens'])
            row = {'task_id': item['task_id'], 'node_id': field['id'], 'original_tokens': item['input_tokens'],
                   'relocated_tokens': replacement['input_tokens'], 'delta': replacement['input_tokens'] - item['input_tokens']}
            lengths.append(row)
            if replacement['input_tokens'] > shared['model']['max_input_tokens']:
                failures.append(row)
            transformed.append(replacement)
        originals.append(cohort)
        changed.append(transformed)
    baseline = ledger(originals, shared['runtime']['root_batch_size'], config['buckets'])
    candidate = ledger(changed, shared['runtime']['root_batch_size'], config['buckets'])
    measured = json.loads(Path(config['original_replay_summary']).read_text())['phases'][config['replay_variant']]['work']
    assert all(baseline['totals'][name] == measured[name] for name in baseline['totals'])
    result = {'source': config['source_run'], 'cohorts': len(originals), 'requests': len(lengths),
              'all_semantic_json_exact': True, 'options_questions_and_task_contexts_unchanged': True,
              'original_ledger_matches_gpu_measured_tokens': True, 'original': baseline, 'relocated': candidate,
              'lengths': lengths, 'over_limit': failures,
              'scope': 'CPU exact token accounting under original cohort order, task/context grouping, last-real-token boundary and cache capacities. Changed conditioning requires model-quality qualification. Overflow requests are retained and not executable under the shared limit.'}
    Path(config['output']).write_text(json.dumps(result, indent=2) + '\n')
    target = Path(config['rewritten_requests'])
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(changed, **config['serialization']) + '\n')
    print(json.dumps({key: result[key] for key in ('cohorts', 'requests', 'all_semantic_json_exact', 'over_limit')}))
    print(json.dumps({'original': baseline['totals'], 'relocated': candidate['totals']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    main(json.loads(parser.parse_args().config.read_text()))
