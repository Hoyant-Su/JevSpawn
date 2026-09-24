from collections import Counter, defaultdict
import json
from pathlib import Path
import time

from transformers import AutoTokenizer

from history_replay import render, coverage
from baselines.common.context_window import truncate_prompt
from jev_spawn.infra.prompts import load_prompt
from jev_spawn.schema import CONTROLLER


def serial(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'))


def measure(tokenizer, strings):
    return dict(occurrences=len(strings), characters=sum(map(len, strings)),
                tokens_separate_sum=sum(map(len, tokenizer(strings, add_special_tokens=False)['input_ids'])))


def analyze(tokenizer, candidate, length):
    source, task, turn, field, rendered = candidate
    components, duplicates = defaultdict(list), defaultdict(list)
    state = json.loads(field['state'])
    for key, value in state.items():
        components['shared.' + key].append(serial(value))
    components['encoded_history'].append(field['history'])
    components['background'].append(field['context'])
    components['all_branch_descriptions'] = [o['description'] for o in field['options']]
    required, feedback = [], []
    for option in field['options']:
        branch = json.loads(option['description'])
        if option['id'] == turn.get('frontier_decision', {}).get('choice'):
            required.extend(branch['state']['current_feedback'])
        for key, value in branch['state'].items():
            text = serial(value)
            components['branch.' + key].append(text)
            duplicates['branch.' + key, text].append('options[' + option['id'] + '].state.' + key)
        for index, record in enumerate(branch['state']['declaration_feedback']):
            feedback.append(record)
            duplicates['feedback_record', serial(record)].append('options[' + option['id'] + '].state.declaration_feedback[' + str(index) + ']')
            for key, value in record.items():
                components['feedback.' + key].append(serial(value))
    repeated = [dict(component=key, occurrences=len(paths), value_characters=len(value),
                     duplicate_characters=(len(paths)-1)*len(value), source_paths=paths[:3],
                     value_preview=value[:120]) for (key, value), paths in duplicates.items()
                if len(paths) > 1 and len(value) > 10]
    repeated.sort(key=lambda r: r['duplicate_characters'], reverse=True)
    policy = json.loads(Path('configs/inference/shared_service.json').read_text())['settings']['input_window']
    tokens = tokenizer(rendered, add_special_tokens=False)['input_ids']
    effective, _, window = truncate_prompt(tokenizer, rendered, tokens, 16384, policy, load_prompt(policy['prompt']))
    revisions = [r['revision'] for r in task['trace']['rounds'] if 'revision' in r]
    completed = [r for r in revisions if 'observation' in r]
    return dict(source=source, task_id=task['task_id'], status=task['status'], turn=turn['turn'],
        prompt_tokens=length, prompt_characters=len(rendered), frontier_options=len(field['options']),
        history_events=len(field['history'].splitlines()),
        components={key: measure(tokenizer, values) for key, values in components.items()},
        exact_duplicate_values=repeated[:12],
        revision_summary=dict(started=len(revisions), completed=len(completed),
            accepted=sum(r['observation']['accepted'] for r in completed),
            generated_tokens=sum(r['generated_tokens'] for r in completed),
            error_counts=dict(Counter(r['observation'].get('error', 'accepted') for r in completed))),
        feedback_records=len(feedback), window=window,
        selected_coverage=coverage(field['history'], effective, required))


def main():
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained('/inspire/hdd/global_public/public_models/Qwen/Qwen3.8-27B', local_files_only=True)
    report = dict(stage='050', longest_frontiers=[], ppnl_failures=[], frontier_request_counts={},
        measurement_notes=['Read-only CPU replay; no model inference or production edits.',
            'Exact longest tokenized frontier over every recorded request in all48stage050tasks.',
            'Component tokenizations are nonadditive; nested feedback components overlap their parents.',
            'Duplicate values may be interned with ordered references; dropping occurrences would lose evidence.'])
    for dataset in ['ppnl', 'lmrlgym_maze', 'textarena_lightsout', 'textarena_rushhour', 'textarena_sokoban', 'korgym_2048']:
        run = Path('runs/native-' + dataset + '-jevspawn-20260923-050')
        labels = json.loads((run / 'session-0000/runtime.json').read_text())['backend']['answer_labels']
        candidates = []
        for source in sorted(run.glob('task-*.json')):
            task = json.loads(source.read_text())
            for turn in task['trace']['rounds']:
                if 'frontier_request' in turn:
                    field = turn['frontier_request']
                    candidates.append((str(source), task, turn, field,
                        render(tokenizer, field, CONTROLLER['prefix_template'], labels)))
        lengths = list(map(len, tokenizer([c[-1] for c in candidates], add_special_tokens=False)['input_ids']))
        index = max(range(len(candidates)), key=lengths.__getitem__)
        row = analyze(tokenizer, candidates[index], lengths[index])
        row['dataset'] = dataset
        report['longest_frontiers'].append(row)
        report['frontier_request_counts'][dataset] = len(candidates)
        if dataset == 'ppnl':
            for task_id in ['ICL_test_set/6', 'ICL_test_set/7']:
                selected = [i for i, c in enumerate(candidates) if c[1]['task_id'] == task_id]
                index = max(selected, key=lengths.__getitem__)
                report['ppnl_failures'].append(analyze(tokenizer, candidates[index], lengths[index]))
        print(dataset, row['prompt_tokens'], flush=True)
    report['cpu_seconds_including_tokenizer_load'] = time.perf_counter() - started
    Path('results/validation/native_stage050_frontier_components_20260923.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
