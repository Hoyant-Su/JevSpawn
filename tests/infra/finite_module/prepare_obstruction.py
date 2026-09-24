import argparse
from copy import deepcopy
import json
from pathlib import Path
import time

from jev_spawn.infra.configuration import resolve_symbol
from jev_spawn.infra.prompts import load_prompt


def prepare(config):
    factory = json.loads(Path(config['factory']).read_text())
    prompts = load_prompt(config['prompts'])
    tasks = [json.loads(line) for line in Path(config['tasks']).read_text().splitlines()]
    output = Path(config['output'])
    output.mkdir(parents=True, exist_ok=True)
    labels = []
    deadline = time.monotonic() + config['timeout_seconds']
    for task in tasks:
        environment = resolve_symbol(factory['class'])(task, {}, output,
            deadline=lambda: deadline - time.monotonic(), **factory['parameters'])
        context = environment.context(False, environment.configuration['serialization'])
        for action in environment.native_actions:
            probe = resolve_symbol(factory['class'])(task, {}, output,
                deadline=lambda: deadline - time.monotonic(), **factory['parameters'])
            before = list(probe.native.position)
            feedback, done = probe.execute(config['tool'], {config['argument']: action})
            blocked = before == list(probe.native.position)
            messages = [{'role': 'system', 'content': prompts['direct']}, {'role': 'user',
                'content': prompts['atomic_obstruction'].format(context=context, action=action)}]
            (output / config['trace_file'].format(index=len(labels))).write_text(
                json.dumps({'messages': messages}, indent=2)+'\n')
            labels.append({'task_id': task['task_id'], 'action': action, 'blocked': blocked,
                           'before': before, 'after': list(probe.native.position), 'feedback': feedback})
    (output / 'labels.json').write_text(json.dumps(labels, indent=2)+'\n')
    print(json.dumps({'tasks': len(tasks), 'fields': len(labels)}))


def prepare_recorded(config):
    factory = json.loads(Path(config['factory']).read_text())
    prompts = load_prompt(config['prompts'])
    tasks = {task['task_id']: task for task in map(json.loads, Path(config['tasks']).read_text().splitlines())}
    output = Path(config['output'])
    output.mkdir(parents=True, exist_ok=True)
    labels = []
    deadline = time.monotonic() + config['timeout_seconds']
    for filename in config['records']:
        record = json.loads(Path(filename).read_text())
        environment = resolve_symbol(factory['class'])(tasks[record['task_id']], {}, output,
            deadline=lambda: deadline - time.monotonic(), **factory['parameters'])
        context = environment.context(False, environment.configuration['serialization'])
        history = []
        for turn, transition in enumerate(record['transitions']):
            state = prompts['environment_state'].format(history=json.dumps(history), current=environment.observation)
            observed = prompts['observed_task'].format(context=context, state=state)
            for action, native_action in environment.native_actions.items():
                native = deepcopy(environment.native)
                before = list(native.position)
                feedback, reward, done = native.step(environment.history + (environment.text_type(native_action, True),))
                messages = [{'role': 'system', 'content': prompts['direct']}, {'role': 'user',
                    'content': prompts['atomic_obstruction'].format(context=observed, action=action)}]
                (output / config['trace_file'].format(index=len(labels))).write_text(
                    json.dumps({'messages': messages}, indent=2)+'\n')
                labels.append({'task_id': record['task_id'], 'turn': turn, 'action': action,
                    'blocked': before == list(native.position), 'before': before, 'after': list(native.position)})
            action = transition['decision']['choice']
            observation, done = environment.execute(config['tool'], {config['argument']: action})
            assert json.loads(observation) == transition['observation']
            history.append({'action': action, 'feedback': json.loads(observation)})
    (output / 'labels.json').write_text(json.dumps(labels, indent=2)+'\n')
    print(json.dumps({'fields': len(labels)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    settings = json.loads(Path(parser.parse_args().config).read_text())
    {'initial': prepare, 'recorded': prepare_recorded}[settings['mode']](settings)
