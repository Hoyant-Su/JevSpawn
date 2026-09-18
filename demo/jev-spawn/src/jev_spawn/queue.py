"""Pack each selected role queue into batches, retaining completed calls on restart."""

import json
from pathlib import Path

from jev_spawn.workflow import Workflow


class QueuedWorkflow(Workflow):
    def batches(self, states, events, method, stage, batch_size):
        for index, start in enumerate(range(0, len(states), batch_size)):
            batch = states[start:start + batch_size]
            path = Path(self.config['run_dir']) / f'queue-{self.rank}-{stage}-{index:04d}.json'
            if path.exists():
                saved = json.loads(path.read_text())
                assert [s['task_id'] for s in saved['states']] == [s['task_id'] for s in batch]
                for state, complete in zip(batch, saved['states']):
                    state.update(complete)
            else:
                batch_events = []
                result = method(batch, batch_events)
                saved = {'states': batch, 'events': batch_events, 'result': result}
                temporary = path.with_suffix('.tmp')
                temporary.write_text(json.dumps(saved, ensure_ascii=False) + '\n')
                temporary.replace(path)
            events.extend(saved['events'])
            yield saved['result']

    def review_fields(self, states, events):
        results = list(self.batches(states, events, super().review_fields, 'review_fields',
                                    self.config['controller_batch_size']))
        return {name: {'choices': [choice for result in results for choice in result[name]['choices']]}
                for name in results[0]}

    def decide(self, states, stage, events):
        method = super().decide
        return [choice for result in self.batches(
            states, events, lambda batch, log: method(batch, stage, log), f'decide-{stage}',
            self.config['controller_batch_size']) for choice in result]

    def spawn(self, states, role, events):
        method = super().spawn
        list(self.batches(states, events, lambda batch, log: method(batch, role, log),
                          f'generate-{role}', self.config['batch_size']))
