import argparse
import json
from pathlib import Path
import time

from transformers import AutoTokenizer
import yaml

from jev_spawn.runtime.state import event_history
from tests.infra.native_state_context.codec import decode_events, encode_events


def run(settings):
    shared = yaml.safe_load(Path(settings['shared_config']).read_text())
    tokenizer = AutoTokenizer.from_pretrained(shared['model']['path'], **settings['tokenizer'])
    rows = []
    for source in settings['sources']:
        original_histories, encoded_histories = [], []
        durations, event_counts = [], []
        tasks = sorted(Path(source).glob('task-*.json'))
        incomplete_rounds = []
        for path in tasks:
            events = {}
            task = json.loads(path.read_text())
            for record in task['trace']['rounds']:
                if 'frontier_request' not in record:
                    incomplete_rounds.append({'task': path.name, 'turn': record['turn']})
                    continue
                original = record['frontier_request']['history']
                assert event_history(events.values()) == original
                started = time.perf_counter()
                encoded = encode_events(events.values(), ())
                durations.append(time.perf_counter() - started)
                assert decode_events(encoded) == list(events.values())
                original_histories.append(original)
                encoded_histories.append(encoded)
                event_counts.append(len(events))
                for trace in record.get('children', {}).values():
                    events.update({event['id']: event for event in trace[-2]['observations']})
        original_tokens = list(map(len, tokenizer(original_histories, add_special_tokens=False)['input_ids']))
        encoded_tokens = list(map(len, tokenizer(encoded_histories, add_special_tokens=False)['input_ids']))
        rows.append({'source': source, 'tasks': len(tasks), 'histories': len(durations),
            'incomplete_rounds': incomplete_rounds, 'roundtrip_exact': True,
            'original_tokens': sum(original_tokens), 'encoded_tokens': sum(encoded_tokens),
            'original_token_counts': original_tokens, 'encoded_token_counts': encoded_tokens,
            'event_counts': event_counts, 'encode_seconds': durations,
            'history_token_reduction': 1 - sum(encoded_tokens) / sum(original_tokens)})
        Path(settings['output']).write_text(json.dumps({'settings': settings, 'rows': rows,
            'scope': 'Exact observation reconstruction on every saved complete turn. History tokens only; not model decisions, correctness, total input reduction or inference speed.'}, indent=2) + '\n')
        print(json.dumps({key: value for key, value in rows[-1].items()
                          if key not in {'original_token_counts', 'encoded_token_counts', 'event_counts', 'encode_seconds'}}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    run(json.loads(parser.parse_args().config.read_text()))
