import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import urllib.request

from environments.maze import native_modules
from jev_spawn.infra.prompts import load_prompt
from project_paths import ROOT


def prepare(configuration):
    settings = json.loads(configuration.read_text())
    runtime = json.loads((ROOT / settings['runtime']).read_text())
    prompts = load_prompt(runtime['prompts'])

    def fetch(item):
        destination = ROOT / item['destination']
        with urllib.request.urlopen(item['url'], timeout=settings['download_timeout_seconds']) as response:
            content = response.read()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
        return destination

    with ThreadPoolExecutor(max_workers=settings['download_workers']) as pool:
        list(pool.map(fetch, settings['sources']))
    identities = json.loads((ROOT / settings['eval_ids']).read_text())
    assert len(identities) == settings['expected_count']
    source = ast.parse((ROOT / settings['reset_source']).read_text())
    positions = next(ast.literal_eval(node.value) for node in ast.walk(source)
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name)
            and target.id == settings['positions_variable'] for target in node.targets))
    factory, _ = native_modules(runtime)
    tasks = []
    for identity in identities:
        game = int(identity['item_id'].removeprefix(settings['item_prefix']))
        start = positions[game]
        environment, _, _ = factory.setup_maze_env(start_position=start, **runtime['environment'])
        history = environment.reset(seed=runtime['seed'], options={'goal': environment.goal, 'init_position': start})
        tasks.append({'task_id': identity['item_id'], 'dataset': settings['dataset'],
            'kind': settings['kind'], 'instruction': prompts['instruction'],
            'input': {'observation': history[runtime['latest_history_index']].text},
            'answer_schema': runtime['answer_schema'],
            'source': {'game': game, 'init_position': list(start), 'goal': environment.goal}})
    destination = ROOT / settings['tasks']
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(''.join(json.dumps(task, **runtime['serialization']) + '\n' for task in tasks))
    print(json.dumps({'dataset': settings['dataset'], 'tasks': len(tasks), 'path': str(destination)}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--configuration', type=Path, required=True)
    prepare(parser.parse_args().configuration)


if __name__ == '__main__':
    main()
