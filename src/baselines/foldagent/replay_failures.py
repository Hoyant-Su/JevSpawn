import argparse
from functools import partial
import json
import importlib.util
from pathlib import Path
import traceback

from transformers import AutoTokenizer



class TraceReplay:
    def __init__(self, tokenizer, task_id, records):
        self.agent_tokenizers = {task_id: tokenizer}
        self.records = records
        self.position = 0

    def complete(self, messages, max_new_tokens, temperature, *, task_id):
        batch, index = self.records[self.position]
        assert batch['task_ids'][index] == task_id
        assert messages == batch['messages'][index], (task_id, self.position, 'message mismatch')
        assert max_new_tokens == batch['max_new_tokens']
        assert temperature == batch['temperature']
        self.position += 1
        return [{'text': batch['texts'][index], 'token_ids': batch['output_token_ids'][index]}]


class ObservedTransport:
    denied = []

    async def create_completion(self, input_ids, *, uid, max_len, messages):
        result = await super().create_completion(input_ids, uid=uid, max_len=max_len, messages=messages)
        if result is None:
            actual = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
            self.denied.append({'input_tokens': len(actual),
                               'history_tokens_remaining': max_len - len(input_ids),
                               'context_tokens_remaining': self.settings['context_length'] - len(actual),
                               'last_message': messages[-1]})
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location('fold_replay_adapter', args.run / 'adapter_at_replay.py')
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    protocol = json.loads((args.run / 'protocol.json').read_text())
    tokenizer = AutoTokenizer.from_pretrained(protocol['native']['model_path'])
    collections = {item['task_id']: item for item in protocol['collections']}
    adapter.ModelTransport = type('ObservedTransport', (ObservedTransport, adapter.ModelTransport), {})
    findings = []
    for block in sorted(args.run.glob('block-*/complete.json')):
        complete = json.loads(block.read_text())
        failed = [row for row in complete['results'] if row.get('error') == "TypeError: argument of type 'NoneType' is not iterable"]
        if not failed:
            continue
        execution = block.parent / complete['attempt'] / 'execution'
        batches = json.loads((execution / 'batches.json').read_text())
        for row in failed:
            identity = row['task_id']
            records = [(batch, batch['task_ids'].index(identity)) for batch in batches if identity in batch['task_ids']]
            replay = TraceReplay(tokenizer, identity, records)
            ObservedTransport.denied = []
            try:
                adapter.solve(collections[identity], partial(replay.complete, task_id=identity),
                              protocol['settings'], protocol['prompts'])
            except TypeError as error:
                frames = traceback.extract_tb(error.__traceback__)
                assert str(error) == "argument of type 'NoneType' is not iterable"
                assert frames[-1].name == 'clean_response'
                assert replay.position == len(records)
                assert ObservedTransport.denied
                findings.append({'task_id': identity, 'replayed_model_calls': replay.position,
                    'last_recorded_input_tokens': records[-1][0]['input_tokens'][records[-1][1]],
                    'denied': ObservedTransport.denied,
                    'traceback': [{'file': frame.filename, 'line': frame.lineno, 'function': frame.name} for frame in frames]})
                print({'task_id': identity, 'calls': replay.position, 'error_site': frames[-1].name,
                       'denied_input_tokens': ObservedTransport.denied[-1]['input_tokens']}, flush=True)
            else:
                raise AssertionError(f'Original failure did not reproduce for {identity}')
    assert len(findings) == 16
    report = {'scope': 'CPU replay of recorded real model responses through the original FoldAgent core. Every submitted prompt, token budget and temperature matches its saved inference request. No new model predictions, benchmark scores or latency measurements.',
              'findings': findings,
              'adapter_snapshot': str(args.run / 'adapter_at_replay.py'),
              'conclusion': 'All16exceptions reproduce in the original core. Legacy transport constrains input plus output instead of the common input-only capacity. The104-query comparison requires a complete corrected rerun.'}
    (args.run / 'failure_replay.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__ == '__main__':
    main()
