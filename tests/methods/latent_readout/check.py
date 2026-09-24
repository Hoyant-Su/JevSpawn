import argparse
import ast
import json
from pathlib import Path

from transformers import AutoTokenizer

from methods.latent_readout.inputs import ROOT, condition_id, load_tasks, preflight, read_rows, render, schedule, validate_settings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    settings = json.loads(args.config.read_text())
    validate_settings(settings)
    prompts = json.loads((ROOT / settings['prompts']).read_text())
    tasks = load_tasks(settings)
    tokenizer = AutoTokenizer.from_pretrained(settings['model_path'], padding_side='left', local_files_only=True)
    records = preflight(tokenizer, tasks, settings, prompts)
    expected = []
    for source in settings['sources']:
        original = read_rows(source['tasks'])[:source['count']]
        expected.extend(original)
    for task, original in zip(tasks, expected):
        assert task['state'] == original['state']
        assert task['field'] == original['fields'][task['field_id']]
        prompt = render(tokenizer, [task], prompts)[0]
        assert task['state'] in prompt and task['field']['question'] in prompt
        assert all(option['id'] in prompt and option['description'] in prompt for option in task['field']['options'])
    pairs = [(task['task_id'], condition_id(condition))
             for _, condition, batch in schedule(tasks, settings, measured=True) for task in batch]
    assert len(pairs) == len(set(pairs)) == len(tasks) * len(settings['conditions'])
    order = [[condition_id(condition) for index, condition, _ in schedule(tasks, settings, True) if index == block]
             for block in range(len(records))]
    for position in range(len(settings['conditions'])):
        assert len({row[position] for row in order}) == len(settings['conditions'])
    partial = tasks[:9]
    assert [len(batch) for _, _, batch in schedule(partial, settings, False)] == [8] * 4 + [1] * 4
    for file in (ROOT / 'src/methods/latent_readout').glob('*.py'):
        ast.parse(file.read_text())
    result = {'cpu_only': True, 'model_loaded': False, 'labels_opened': False,
              'original_roots': len(tasks), 'measured_task_condition_pairs': len(pairs),
              'native_model': settings['model_path'], 'conditions': settings['conditions'],
              'batches': records, 'measured_condition_order': order,
              'maximum_physical_history': max(record['maximum_physical_history'] for record in records),
              'original_state_and_options_retained': True, 'id_and_state_disjoint_from_formal_sets': True,
              'candidate_token_boundaries_exact': True, 'partial_batches_not_duplicated': True}
    assert not args.output.exists()
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
