import argparse
from itertools import groupby
import json
from pathlib import Path

from transformers import AutoTokenizer

from baselines.common.config import SharedConfig
from jev_spawn.algo.structured import common_prefix
from jev_spawn.infra.readout_labels import native_labels
from jev_spawn.schema import controller_prefix


def prepare(settings):
    shared = SharedConfig.load(settings['shared_config'])
    tokenizer = AutoTokenizer.from_pretrained(shared.model.path, local_files_only=True)
    labels, label_ids = native_labels(tokenizer, json.loads(Path(settings['labels']).read_text()))
    workloads, pending = [], []
    for source in settings['sources']:
        config = json.loads(Path(source['config']).read_text())
        run = Path(config['run_output'])
        trace_path = run / settings['task_file']
        if not trace_path.exists():
            pending.append(source['track'])
            continue
        trace = json.loads(trace_path.read_text())
        turns = [turn for turn in trace['trace']['rounds'] if turn.get('children') and
                 any('fields' in record for records in turn['parent_computations'].values() for record in records)]
        assert turns, str(trace_path)
        turn = turns[settings['complete_turn_index']]
        batches, count = [], 0
        for parent, records in turn['parent_computations'].items():
            records = [record for record in records if 'fields' in record]
            for field_id, grouped in groupby(records, key=lambda record: record['fields'][0]['id']):
                packets = []
                for record in grouped:
                    field, = record['requests']
                    definition, = record['fields']
                    assert field['id'] == field_id and len(field['options']) == len(definition['values'])
                    messages = field['action_messages']
                    rendered = tokenizer.apply_chat_template(messages, **settings['chat_template'])
                    tokens = tokenizer(rendered, add_special_tokens=False)['input_ids']
                    root_messages = [messages[0], {'role': 'user', 'content': controller_prefix(field['context'], '')}]
                    root = tokenizer.apply_chat_template(root_messages, **settings['chat_template'])
                    root_tokens = tokenizer(root, add_special_tokens=False)['input_ids']
                    root_length = common_prefix([tokens, root_tokens])
                    assert len(tokens) <= shared.model.max_input_tokens
                    packets.append({'field': field, 'definition': definition, 'messages': messages,
                        'rendered': rendered, 'tokens': tokens,
                        'root_tokens': tokens[:root_length], 'task_id': trace['task_id'],
                        'candidate_token_ids': label_ids[:len(field['options'])]})
                for start in range(0, len(packets), shared.runtime.branch_batch_size):
                    batch = packets[start:start + shared.runtime.branch_batch_size]
                    batches.append({'parent': parent, 'field_id': field_id, 'requests': batch})
                    count += len(batch)
        assert count == sum(len(record['requests']) for records in turn['parent_computations'].values()
                            for record in records if 'fields' in record)
        workloads.append({'track': source['track'], 'source': str(trace_path), 'turn': turn['turn'],
            'task_id': trace['task_id'], 'spawned_children': turn['children'],
            'protocol': str(run / 'protocol.json'), 'finite_request_count': count,
            'batches': batches})
    report = {'settings': settings, 'pending_tracks': pending, 'workloads': workloads,
        'input_equality': 'Both modes consume each identical prepared packet; no prompt rewriting or branch resampling.',
        'measurement_scope': 'All model-dependent action-field readouts from one complete recorded spawn turn. '
            'Native tool execution, controller decisions, declaration generation and beam selection are excluded.',
        'batching': 'Consecutive requests for the same parent and field remain together, sharded only at shared branch_batch_size. '
            'This deterministic isolated-turn replay does not reproduce cross-sample online queue interleaving.'}
    path = Path(settings['prepared'])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'prepared': len(workloads), 'pending': pending,
        'requests': {workload['track']: workload['finite_request_count'] for workload in workloads}}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    prepare(json.loads(parser.parse_args().config.read_text()))
