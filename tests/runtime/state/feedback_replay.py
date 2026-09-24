from copy import deepcopy
import json
from pathlib import Path
import time

from transformers import AutoTokenizer

from history_replay import coverage, render, task_events
from baselines.common.context_window import truncate_prompt
from jev_spawn.infra.prompts import load_prompt
from jev_spawn.runtime.state import history_frontier
from jev_spawn.schema import CONTROLLER


def serial(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'))


def replace_feedback(field, events):
    state = json.loads(field['state'])
    branches = {o['id']: json.loads(o['description']) for o in field['options']}
    original = deepcopy(branches)
    for branch in branches.values():
        branch['state']['computed_fields'] = [{**state['field_definitions'][index], 'value': value}
            for index, value in branch['state'].pop('computed_values')]
    raw = dict(execution_events=[events[i] for i in state['execution_order']],
               declarations=state['declarations'], branches=branches)
    shared, packed = history_frontier(raw)
    restored = deepcopy(packed)
    for branch in restored.values():
        item = branch['state']
        if 'declaration_feedback_ids' in item:
            item['declaration_feedback'] = [deepcopy(shared['declaration_feedback_records'][index])
                for index in item.pop('declaration_feedback_ids')]
    assert json.dumps(restored, sort_keys=True) == json.dumps(original, sort_keys=True)
    before, after = deepcopy(state), deepcopy(shared)
    before.pop('format')
    after.pop('format')
    after.pop('declaration_feedback_records')
    assert json.dumps(before, sort_keys=True) == json.dumps(after, sort_keys=True)
    new = deepcopy(field)
    new['state'] = serial(shared)
    new['options'] = [dict(option, description=serial(packed[option['id']])) for option in field['options']]
    return new, len(shared['declaration_feedback_records'])


def main():
    started = time.perf_counter()
    prior = json.loads(Path('results/validation/native_stage050_frontier_components_20260923.json').read_text())
    targets = {(r['source'], r['turn']) for r in prior['longest_frontiers'] + prior['ppnl_failures']}
    measured, requests, branches = [], 0, 0
    for dataset in prior['frontier_request_counts']:
        run = Path('runs/native-' + dataset + '-jevspawn-20260923-050')
        labels = json.loads((run / 'session-0000/runtime.json').read_text())['backend']['answer_labels']
        for source in sorted(run.glob('task-*.json')):
            task = json.loads(source.read_text())
            events = {event['id']: event for event in task_events(task)[0]}
            for turn in task['trace']['rounds']:
                if 'frontier_request' not in turn:
                    continue
                old = turn['frontier_request']
                before = serial(old)
                new, unique = replace_feedback(old, events)
                assert serial(old) == before
                requests += 1
                branches += len(old['options'])
                if (str(source), turn['turn']) in targets:
                    selected = turn.get('frontier_decision', {}).get('choice')
                    required = [i for option in old['options'] if option['id'] == selected
                        for i in json.loads(option['description'])['state']['current_feedback']]
                    measured.append((source, task['task_id'], turn['turn'], old, new, unique, labels, required))
    structural_seconds = time.perf_counter() - started
    print(json.dumps(dict(frontiers=requests, branches=branches, exact_roundtrip=True,
                          structural_seconds=structural_seconds)), flush=True)
    tokenizer = AutoTokenizer.from_pretrained('/inspire/hdd/global_public/public_models/Qwen/Qwen3.8-27B', local_files_only=True)
    policy = json.loads(Path('configs/inference/shared_service.json').read_text())['settings']['input_window']
    rows = []
    for source, task_id, turn, old, new, unique, labels, required in measured:
        texts = [render(tokenizer, field, CONTROLLER['prefix_template'], labels) for field in [old, new]]
        encoded = tokenizer(texts, add_special_tokens=False)['input_ids']
        counts = list(map(len, encoded))
        availability = {}
        for name, text, tokens, field in zip(['old', 'new'], texts, encoded, [old, new], strict=True):
            effective, _, metadata = truncate_prompt(tokenizer, text, tokens, 16384, policy, load_prompt(policy['prompt']))
            availability[name] = dict(window=metadata, selected_events=coverage(field['history'], effective, required))
        rows.append(dict(source=str(source), task_id=task_id, turn=turn,
            options=len(old['options']), unique_feedback_records=unique,
            old_prompt_tokens=counts[0], new_prompt_tokens=counts[1],
            old_prompt_characters=len(texts[0]), new_prompt_characters=len(texts[1]),
            old_exceeds_window=counts[0] > 16384, new_exceeds_window=counts[1] > 16384,
            retained_evidence=availability))
    report = dict(stage='052', frontiers=requests, branches=branches, exact_roundtrip=True,
        source_unchanged=True, structural_seconds=structural_seconds,
        total_cpu_seconds_including_tokenizer_load=time.perf_counter()-started,
        protocol=dict(table='declaration_feedback_records', references='declaration_feedback_ids'),
        protocol_text=load_prompt('jevspawn.state')['history_format'],
        old_truncated=sum(row['old_exceeds_window'] for row in rows),
        new_truncated=sum(row['new_exceeds_window'] for row in rows),
        measurement_notes=['All804recorded050frontiers roundtrip exactly apart from protocol metadata.',
            'Real tokenizer measurements use the six prior maximum frontiers plus PPNL failures6/7.',
            'Old and new use the same current controller wrapper and recorded candidate labels. New state includes052protocol text.',
            'No model computation or planning-quality claim. Remaining overflow is explicit.'], rows=rows)
    Path('results/validation/native_stage052_feedback_replay_20260923.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(rows), flush=True)


if __name__ == '__main__':
    main()
