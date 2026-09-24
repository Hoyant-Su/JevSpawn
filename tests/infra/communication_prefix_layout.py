import argparse
from collections import defaultdict
from copy import deepcopy
import json
from pathlib import Path

from transformers import AutoTokenizer
import yaml

from baselines.common.resources import TEMPLATES
from jev_spawn.infra.configuration import load_resource
from jev_spawn.infra.prompts import load_prompt
from jev_spawn.infra.readout_labels import native_labels
from jev_spawn.runtime.reference_transport import unpack_references
from jev_spawn.schema import CONTROLLER, controller_prompts


def prefix_length(sequences):
    return next((index for index, values in enumerate(zip(*sequences)) if len(set(values)) != 1),
                min(map(len, sequences)))


def grouped_tokens(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row['task_id'], row['field']['context'])].append(row['tokens'])
    prefixes = [prefix_length(sequences) for sequences in groups.values()]
    return {'logical_tokens': sum(len(row['tokens']) for row in rows),
            'shared_tokens': sum(length * (len(sequences) - 1)
                                 for length, sequences in zip(prefixes, groups.values(), strict=True)),
            'root_prefix_lengths': prefixes}


def main(config):
    shared = yaml.safe_load(Path(config['shared_config']).read_text())
    transport = load_resource(config['transport'])
    prompts = load_prompt(config['prompts'])
    tokenizer = AutoTokenizer.from_pretrained(shared['model']['path'], local_files_only=True)
    labels, _ = native_labels(tokenizer, load_resource(config['labels']))
    protocol = json.loads(Path(config['source_run'], 'protocol.json').read_text())
    CONTROLLER['option_template'] = protocol['prompts']['option_template']
    original = [json.loads(path.read_text()) for path in sorted(Path(config['source_run']).glob(config['input_glob']))]
    cohorts, messages, originals = [], [], []
    for cohort in original:
        rewritten = []
        for row in cohort['requests']:
            field = deepcopy(row['field'])
            decoded = unpack_references(field['input'], transport)
            ordered = {key: decoded['input'][key] for key in config['shared_keys'] if key in decoded['input']}
            ordered.update(decoded['input'])
            literal = {**decoded, 'input': ordered}
            assert literal == decoded
            field['input'] = {transport['root_key']: literal}
            state = prompts['encoded_state'].format(value=json.dumps(field['input'], **config['serialization']))
            field['state'] = TEMPLATES['worker_state'].format(context=field['context'], state=state)
            field['options'] = [{'id': option['id'], 'description': prompts['reference_option'].format(index=index)}
                                for index, option in enumerate(field['options'])]
            user, = controller_prompts([field['state']], field['question'], field['options'],
                labels[:len(field['options'])], CONTROLLER['output_instruction'])
            current = [{'role': 'system', 'content': CONTROLLER['system']}, {'role': 'user', 'content': user}]
            item = {**row, 'field': field, 'messages': current}
            rewritten.append(item)
            messages.append(current)
            originals.append(row)
        cohorts.append({'cohort': cohort['cohort'], 'requests': rewritten})
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                              enable_thinking=shared['generation']['enable_thinking'])
    encoded = tokenizer(rendered, add_special_tokens=False)['input_ids']
    rows = [row for cohort in cohorts for row in cohort['requests']]
    for old, row, text, tokens in zip(originals, rows, rendered, encoded, strict=True):
        assert old['tokens'][:old['base_length']] == tokens[:old['base_length']]
        row.update(rendered=text, tokens=tokens, input_tokens=len(tokens))
    per_cohort = [{'cohort': old['cohort'], 'original': grouped_tokens(old['requests']),
                  'candidate': grouped_tokens(new['requests'])}
                  for old, new in zip(original, cohorts, strict=True)]
    report = {'configuration': config, 'requests': len(rows), 'cohorts': per_cohort,
        'original_logical_tokens': sum(len(row['tokens']) for row in originals),
        'candidate_logical_tokens': sum(len(row['tokens']) for row in rows),
        'original_within_cohort_reused_tokens': sum(row['original']['shared_tokens'] for row in per_cohort),
        'candidate_within_cohort_reused_tokens': sum(row['candidate']['shared_tokens'] for row in per_cohort),
        'maximum_input_tokens': max(len(row['tokens']) for row in rows),
        'overflow': [{'task_id': row['task_id'], 'node': row['field']['id'], 'tokens': len(row['tokens'])}
                     for row in rows if len(row['tokens']) > shared['model']['max_input_tokens']],
        'exact_semantics_and_task_prefix': True, 'gpu_executed': False,
        'scope': config['scope']}
    Path(config['records']).write_text(json.dumps(cohorts, **config['serialization']) + '\n')
    Path(config['output']).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: value for key, value in report.items() if key not in {'configuration', 'cohorts'}}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    main(json.loads(parser.parse_args().config.read_text()))
