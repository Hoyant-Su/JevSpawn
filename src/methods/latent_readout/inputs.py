import json
import string
from pathlib import Path
from jev_spawn.infra.prompts import resolve_prompts


ROOT = Path(__file__).resolve().parents[3]


def validate_settings(settings):
    stage = resolve_prompts(json.loads((ROOT / 'configs/methods/structured_flow/latent-readout-stage.json').read_text()))
    for key in ['model_path', 'dtype', 'batch_size', 'max_input_tokens', 'seed', 'readout_suffix']:
        assert settings[key] == stage['fixed'][key]
    assert settings['conditions'] == stage['conditions']
    assert settings['task_count'] == sum(source['count'] for source in settings['sources']) == 32


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def load_tasks(settings):
    tasks = []
    for source in settings['sources']:
        rows = read_rows(source['tasks'])
        assert len(rows) >= source['count']
        selected = rows[:source['count']]
        evaluation = read_rows(source['evaluation_tasks'])
        assert not ({row['task_id'] for row in selected} & {row['task_id'] for row in evaluation})
        assert not ({row['state'] for row in selected} & {row['state'] for row in evaluation})
        for row in selected:
            assert len(row['fields']) == 1
            field_id, field = next(iter(row['fields'].items()))
            assert 2 <= len(field['options']) <= len(string.ascii_uppercase)
            assert len({option['id'] for option in field['options']}) == len(field['options'])
            tasks.append({'task_id': row['task_id'], 'dataset': source['dataset'],
                          'state': row['state'], 'field_id': field_id, 'field': field})
    assert len(tasks) == settings['task_count']
    assert len({task['task_id'] for task in tasks}) == len(tasks)
    return tasks


def render(tokenizer, tasks, prompts):
    messages = []
    for task in tasks:
        menu = '\n'.join(prompts['option'].format(label=label, **option)
                         for label, option in zip(string.ascii_uppercase, task['field']['options']))
        user = prompts['user'].format(state=task['state'], question=task['field']['question'], menu=menu)
        messages.append([{'role': 'system', 'content': prompts['system']},
                         {'role': 'user', 'content': user}])
    return tokenizer.apply_chat_template(messages, tokenize=False,
                                        add_generation_prompt=True, enable_thinking=False)


def candidate_ids(tokenizer, count, suffix):
    labels = list(string.ascii_uppercase[:count])
    ids = tokenizer(labels, add_special_tokens=False)['input_ids']
    assert all(len(value) == 1 for value in ids)
    assert len({value[0] for value in ids}) == count
    suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
    assert tokenizer([suffix + label for label in labels], add_special_tokens=False)['input_ids'] == [
        suffix_ids + value for value in ids], 'Candidate tokenization changes at the suffix boundary.'
    return [value[0] for value in ids]


def preflight(tokenizer, tasks, settings, prompts):
    suffix_ids = tokenizer.encode(settings['readout_suffix'], add_special_tokens=False)
    maximum_work = max(condition['steps'] if condition['mode'] == 'latent'
                       else condition['max_new_tokens'] for condition in settings['conditions'])
    records = []
    for start in range(0, len(tasks), settings['batch_size']):
        batch = tasks[start:start + settings['batch_size']]
        texts = render(tokenizer, batch, prompts)
        encoded = tokenizer(texts, padding=True, truncation=False, add_special_tokens=False)
        width = len(encoded['input_ids'][0])
        lengths = [sum(mask) for mask in encoded['attention_mask']]
        assert all(len(ids) == width for ids in encoded['input_ids'])
        assert width + maximum_work + len(suffix_ids) <= settings['max_input_tokens']
        for task, text in zip(batch, texts):
            ids = candidate_ids(tokenizer, len(task['field']['options']), settings['readout_suffix'])
            prefix = text + settings['readout_suffix']
            prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
            for letter, token_id in zip(string.ascii_uppercase, ids):
                assert tokenizer.encode(prefix + letter, add_special_tokens=False) == prefix_ids + [token_id]
        records.append({'task_ids': [task['task_id'] for task in batch], 'batch_size': len(batch),
                        'prompt_tokens': lengths, 'prompt_width': width,
                        'suffix_tokens': len(suffix_ids), 'maximum_physical_history': width + maximum_work + len(suffix_ids),
                        'candidate_counts': [len(task['field']['options']) for task in batch]})
    return records


def condition_id(condition):
    return f"latent-{condition['steps']}" if condition['mode'] == 'latent' else 'text'


def schedule(tasks, settings, measured):
    conditions = settings['conditions']
    for block, start in enumerate(range(0, len(tasks), settings['batch_size'])):
        offset = block % len(conditions) if measured else 0
        for condition in conditions[offset:] + conditions[:offset]:
            yield block, condition, tasks[start:start + settings['batch_size']]
