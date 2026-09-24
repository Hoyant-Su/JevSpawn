import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import json
from pathlib import Path
import time

from transformers import AutoTokenizer

from baselines.common.context_window import truncate_prompt
from jev_spawn.infra.prompts import load_prompt
from jev_spawn.runtime.state import event_history, extend_event_history
from jev_spawn.schema import CONTROLLER


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def value_at(value, path):
    for key in path:
        value = value[key]
    return value


def restore(records):
    restored = {}
    for record in records:
        event = deepcopy(record)
        for reference in event.pop('text_prefixes', []):
            parent = restored[reference['event']]['value']
            prefix = ''.join(value_at(parent, reference['path']).splitlines(keepends=True)[:reference['lines']])
            target = event
            path = ['value', *reference['path']]
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = prefix + target[path[-1]]
        assert all(identity in restored for identity in event['parent_feedback'])
        restored[event['id']] = event
    return list(restored.values())


def task_events(task):
    seen, groups = {}, []
    for turn in task['trace']['rounds']:
        group = []
        for child in turn.get('children', {}).values():
            for step in child:
                for event in step.get('observations', []):
                    identity = event['id']
                    if identity in seen:
                        assert canonical(seen[identity]) == canonical(event)
                    else:
                        seen[identity] = event
                        group.append(event)
        groups.append(group)
    return list(seen.values()), groups


def verify_task(source):
    task = json.loads(Path(source).read_text())
    events, groups = task_events(task)
    original = canonical(events)
    started = time.perf_counter()
    encoded = event_history(events)
    encode_seconds = time.perf_counter() - started
    records = [json.loads(line) for line in encoded.splitlines()]
    assert canonical(restore(records)) == original, source
    preceding, incremental = [], ''
    started = time.perf_counter()
    for group in groups:
        incremental += extend_event_history(group, preceding)
        preceding.extend(group)
        assert encoded.startswith(incremental), source
    assert incremental == encoded and canonical(events) == original, source
    incremental_seconds = time.perf_counter() - started
    serialization = load_prompt('jevspawn.state')['history_serialization']
    full = ''.join(json.dumps(event, **serialization) + '\n' for event in events)
    return dict(source=source, task_id=task['task_id'], events=len(events),
                prefix_references=sum(len(r.get('text_prefixes', [])) for r in records),
                original_characters=len(full), encoded_characters=len(encoded),
                encode_seconds=encode_seconds, incremental_seconds=incremental_seconds,
                exact_roundtrip=True, stable_append=True, source_unchanged=True)


def current_format(value):
    if isinstance(value, dict):
        return {key: load_prompt('jevspawn.state')['history_format'] if key == 'format'
                else current_format(child) for key, child in value.items()}
    if isinstance(value, list):
        return [current_format(child) for child in value]
    return value


def render(tokenizer, field, old_prefix, labels):
    menu = '\n'.join(CONTROLLER['option_template'].format(label=label, **option)
                     for label, option in zip(labels[:len(field['options'])], field['options'], strict=True))
    prefix = old_prefix.format(context=field['context'], history=field['history'])
    prompt = CONTROLLER['user_template'].format(prefix=prefix, context=field['context'],
        state=field['state'], question=field['question'], menu=menu,
        output_instruction=CONTROLLER['output_instruction'])
    return tokenizer.apply_chat_template([
        {'role': 'system', 'content': CONTROLLER['system']},
        {'role': 'user', 'content': prompt}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False)


def coverage(history, effective, required):
    available, retained = {}, {}
    for line in history.splitlines():
        record = json.loads(line)
        identity = record['id']
        retained[identity] = line in effective
        available[identity] = retained[identity] and all(
            available[ref['event']] for ref in record.get('text_prefixes', []))
    return dict(required_ids=required, complete_encoded_records=sum(retained[i] for i in required),
                fully_reconstructable=sum(available[i] for i in required),
                unavailable_ids=[i for i in required if not available[i]])


def prompt_measurements(config, tokenizer):
    policy = json.loads(Path(config['inference']).read_text())['settings']['input_window']
    notice = load_prompt(policy['prompt'])
    output = []
    for track in config['tracks']:
        if not track['measure_prompts']:
            continue
        for source in track['sources']:
            task = json.loads(Path(source).read_text())
            runtime = json.loads((Path(source).parent / 'session-0000/runtime.json').read_text())
            labels = runtime['backend']['answer_labels']
            for kind in ['operation', 'frontier']:
                turn = next(t for t in reversed(task['trace']['rounds']) if kind + '_request' in t)
                old = turn[kind + '_request']
                new = deepcopy(old)
                events = [json.loads(line) for line in old['history'].splitlines()]
                new['history'] = event_history(events)
                new['state'] = json.dumps(current_format(json.loads(old['state'])), separators=(',', ':'), ensure_ascii=False)
                new['options'] = [dict(option, description=json.dumps(current_format(json.loads(option['description'])),
                    separators=(',', ':'), ensure_ascii=False)) for option in old['options']] if kind == 'frontier' else old['options']
                if kind == 'operation':
                    required = json.loads(old['state'])['state']['current_feedback']
                    all_required = required
                else:
                    all_required = [identity for option in old['options']
                        for identity in json.loads(option['description'])['state']['current_feedback']]
                    selected = turn.get('frontier_decision', {}).get('choice')
                    required = [identity for option in old['options'] if option['id'] == selected
                        for identity in json.loads(option['description'])['state']['current_feedback']]
                row = dict(dataset=track['dataset'], task_id=task['task_id'], status=task['status'],
                    source=source, kind=kind, turn=turn['turn'], model_required=len(old['options']) > 1)
                for name, field, prefix in [('old', old, config['old_prefix']), ('new', new, CONTROLLER['prefix_template'])]:
                    rendered = render(tokenizer, field, prefix, labels)
                    tokens = tokenizer(rendered, add_special_tokens=False)['input_ids']
                    effective, _, metadata = truncate_prompt(tokenizer, rendered, tokens, config['max_input_tokens'], policy, notice)
                    row[name] = dict(**metadata, selected_coverage=coverage(field['history'], effective, required),
                                     frontier_coverage=coverage(field['history'], effective, all_required))
                output.append(row)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    sources = [source for track in config['tracks'] for source in track['sources']]
    assert len(sources) == config['expected_tasks'] == len(set(sources))
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=config['cpu_workers']) as pool:
        tasks = list(pool.map(verify_task, sources))
    structural_seconds = time.perf_counter() - started
    print(json.dumps(dict(stage='structural_complete', tasks=len(tasks), events=sum(t['events'] for t in tasks),
        seconds=structural_seconds, original_characters=sum(t['original_characters'] for t in tasks),
        encoded_characters=sum(t['encoded_characters'] for t in tasks))), flush=True)
    tokenizer = AutoTokenizer.from_pretrained(config['model'], local_files_only=True)
    started = time.perf_counter()
    prompts = prompt_measurements(config, tokenizer)
    report = dict(stage='050', config=args.config, tasks=len(tasks),
        structural_seconds=structural_seconds, prompt_measurement_seconds=time.perf_counter()-started,
        exact_roundtrip=True, stable_append=True, source_unchanged=True,
        reconstruction_note=config['reconstruction_note'], event_checks=tasks, prompt_checks=prompts)
    report['summary'] = []
    for track in config['tracks']:
        checks = [p for p in prompts if p['dataset'] == track['dataset']]
        if checks:
            report['summary'].append(dict(dataset=track['dataset'], requests=len(checks),
                old_tokens=sum(p['old']['original_tokens'] for p in checks),
                new_tokens=sum(p['new']['original_tokens'] for p in checks),
                old_truncated=sum(p['old']['omitted_tokens'] > 0 for p in checks),
                new_truncated=sum(p['new']['omitted_tokens'] > 0 for p in checks),
                old_selected_missing=sum(bool(p['old']['selected_coverage']['unavailable_ids']) for p in checks),
                new_selected_missing=sum(bool(p['new']['selected_coverage']['unavailable_ids']) for p in checks)))
    Path(config['output']).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report['summary']), flush=True)


if __name__ == '__main__':
    main()
