import argparse
from copy import deepcopy
import json
from pathlib import Path

from jev_spawn.runtime.state import pack_state


def unpack_state(packed):
    state = deepcopy(packed)
    observations = state.pop('observations')
    definitions = state.pop('field_definitions')
    state.pop('format')
    for event in state['execution_events']:
        event['value'] = deepcopy(observations[event.pop('observation_id')])
    state['computed_fields'] = [dict(deepcopy(definitions[index]), value=deepcopy(value))
                                for index, value in state.pop('computed_values')]
    assert pack_state(state) == packed
    return state


def extract(configuration):
    manifest = []
    for entry in configuration['sources']:
        source = Path(entry['source'])
        batch = json.loads(source.read_text())[entry['batch_index']]
        row = entry['row_index']
        assert batch['task_ids'][row] == entry['task_id']
        messages = batch['messages'][row]
        payload = json.loads(messages[-1]['content'])
        state = unpack_state(payload['actual_state'])
        fixture = {'task_id': entry['task_id'], 'context': payload['context'],
                   'state': state, 'answer_schema': payload['answer_schema'],
                   'selected_action_history': payload['selected_action_history']}
        path = Path(entry['fixture_path'])
        path.write_text(json.dumps(fixture, indent=2) + '\n')
        manifest.append({**entry, 'state_roundtrip_equal': True,
                         'execution_events': len(state['execution_events']),
                         'selected_path_events': len(state['current_path'])})
    output = {'scope': configuration['scope'], 'fixtures': manifest}
    Path(configuration['manifest']).write_text(json.dumps(output, indent=2) + '\n')
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    result = extract(json.loads(Path(args.config).read_text()))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
