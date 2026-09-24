import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

from tokenizers import Tokenizer

from jev_spawn.algo.structured import common_prefix
from jev_spawn.runtime.state import frontier_view, pack_state
from jev_spawn.schema import CONTROLLER, controller_prompts
from test_state_packing import unpack, unpack_view


parser = argparse.ArgumentParser()
parser.add_argument('--manifest', type=Path, required=True)
parser.add_argument('--tokenizer', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
manifest = json.loads(args.manifest.read_text())
tokenizer = Tokenizer.from_file(str(args.tokenizer))
texts, rows, unfinished, cohorts = [], [], [], []
for run in manifest['runs']:
    if run['method'] != 'jevspawn':
        continue
    directory = Path(run['run'])
    protocol = json.loads((directory / 'protocol.json').read_text())
    serialization = protocol['method']['settings']['rollout']['execution']['serialization']
    CONTROLLER['option_template'] = protocol['prompts']['option_template']
    runtime = json.loads((directory / 'session-0000/runtime.json').read_text())
    labels = runtime['service']['candidate_labels']
    for artifact in sorted(directory.glob('task-*.json')):
        task = json.loads(artifact.read_text())
        for record in task['trace']['rounds']:
            for parent, computations in record.get('parent_computations', {}).items():
                groups = defaultdict(list)
                for computation in computations:
                    for request in computation.get('requests', []):
                        groups[request['id']].append(request)
                for field, requests in groups.items():
                    if len(requests) < 2:
                        continue
                    layouts = [[], []]
                    for request in requests:
                        original = unpack(json.loads(request['state']), frontier=False)
                        packed = pack_state(original)
                        assert unpack(packed, frontier=False) == original
                        updated = {**request, 'state': json.dumps(packed, **serialization)}
                        for output, value in zip(layouts, [request, updated], strict=True):
                            output.append(controller_prompts([value['state']], value['question'], value['options'],
                                labels[:len(value['options'])], CONTROLLER['output_instruction'],
                                contexts=[value['context']])[0])
                    encoded_layouts = [[encoding.ids for encoding in tokenizer.encode_batch(layout, add_special_tokens=False)]
                                       for layout in layouts]
                    prefix_lengths = [common_prefix(layout) for layout in encoded_layouts]
                    cohorts.append({'task_id': task['task_id'], 'turn': record['turn'], 'parent': parent,
                        'field': field, 'rows': len(requests), 'original_common_prefix_tokens': prefix_lengths[0],
                        'updated_common_prefix_tokens': prefix_lengths[1],
                        'original_suffix_tokens': sum(len(row) - prefix_lengths[0] for row in encoded_layouts[0]),
                        'updated_suffix_tokens': sum(len(row) - prefix_lengths[1] for row in encoded_layouts[1]),
                        'exact_reconstruction': True})
            if 'frontier_request' not in record:
                unfinished.append({'task_id': task['task_id'], 'turn': record['turn']})
                continue
            request = record['frontier_request']
            state = json.loads(request['state'])
            branches = {option['id']: json.loads(option['description']) for option in request['options']}
            original = unpack_view(state, branches)
            started = time.perf_counter()
            packed, choices = frontier_view(original)
            seconds = time.perf_counter() - started
            assert unpack_view(packed, choices) == original
            updated = {**request, 'state': json.dumps(packed, **serialization),
                       'options': [{'id': identity, 'description': json.dumps(value, **serialization)}
                                   for identity, value in choices.items()]}
            for value in [request, updated]:
                texts.append(controller_prompts([value['state']], value['question'], value['options'],
                    labels[:len(value['options'])], CONTROLLER['output_instruction'],
                    contexts=[value['context']])[0])
            rows.append({'task_id': task['task_id'], 'turn': record['turn'],
                         'events': len(original['execution_events']), 'branches': len(branches),
                         'exact_reconstruction': True, 'packing_seconds': seconds})
encoded = tokenizer.encode_batch(texts, add_special_tokens=False)
for index, row in enumerate(rows):
    row.update(original_user_tokens=len(encoded[index * 2].ids),
               compact_user_tokens=len(encoded[index * 2 + 1].ids))
report = {'scope': 'Exact observation reconstruction and decision user-message token counts. Model system and chat wrapper overhead is excluded equally from both counts.', 'rows': rows, 'conditional_cohorts': cohorts, 'unfinished_rounds_without_recorded_request': unfinished}
args.output.write_text(json.dumps(report, indent=2) + '\n')
for task_id in dict.fromkeys(row['task_id'] for row in rows):
    selected = [row for row in rows if row['task_id'] == task_id]
    print(json.dumps({'task_id': task_id, 'rounds': len(selected), 'last': selected[-1]}))
