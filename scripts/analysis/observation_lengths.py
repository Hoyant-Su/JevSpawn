import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

from tokenizers import Tokenizer
import yaml


def measure(configuration):
    shared = yaml.safe_load(Path(configuration['shared_config']).read_text())
    tokenizer = Tokenizer.from_file(str(Path(shared['model']['path']) / 'tokenizer.json'))
    limits = configuration['candidate_limits']

    def sample(path):
        record = json.loads(path.read_text())
        trace = record.get('trace') or record
        actions = trace.get('actions', [])
        observations = [action['result']['observation'] if 'observation' in action['result']
                        else action['result'] for action in actions]
        texts = [value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
                 for value in observations]
        lengths = [len(value.ids) for value in tokenizer.encode_batch(texts, add_special_tokens=False)]
        return {'task_id': record['task_id'], 'status': record['status'],
                'recorded_observations': len(lengths), 'tokens': lengths,
                'exceeding': {str(limit): sum(length > limit for length in lengths) for limit in limits}}

    results = {}
    with ThreadPoolExecutor(max_workers=shared['runtime']['cpu_threads']) as pool:
        for dataset in configuration['datasets']:
            paths = sorted((Path(configuration['run']) / dataset).glob('task-*.json'))
            rows = list(pool.map(sample, paths))
            lengths = [length for row in rows for length in row['tokens']]
            results[dataset] = {'recorded_tasks': len(rows),
                'tasks_without_recorded_observations': sum(not row['tokens'] for row in rows),
                'observations': len(lengths), 'max_tokens': max(lengths) if lengths else None,
                'exceeding': {str(limit): sum(row['exceeding'][str(limit)] for row in rows) for limit in limits},
                'tasks': rows}
    output = {'scope': 'Recorded ReAct observations; absent traces are not evidence of short observations.',
              'configuration': configuration, 'results': results}
    Path(configuration['output']).write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps({name: {key: value for key, value in result.items() if key != 'tasks'}
                      for name, result in results.items()}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=Path)
    measure(json.loads(parser.parse_args().config.read_text()))
