import argparse
from collections import Counter, deque
from copy import deepcopy
import json
from pathlib import Path
import time

import yaml

from declaration_builder_stage054 import DeclarationBuilder as ReferenceBuilder
from jev_spawn.infra.prompts import load_prompt
from jev_spawn.schema.declaration import compile_declaration
from jev_spawn.schema.declaration_builder import DeclarationBuilder


ROOT = Path(__file__).resolve().parents[4]


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


class RecordedService:
    def __init__(self, calls, task_id):
        self.calls = deque(call for call in calls if call['kind'] in ('finite', 'signature'))
        self.task_id = task_id

    def decide(self, requests, *, task_id):
        call = self.calls.popleft()
        assert call['kind'] == 'finite' and task_id == self.task_id
        assert [r['options'] for r in requests] == [r['options'] for r in call['requests']]
        return deepcopy(call['outputs'])

    def complete_batch(self, messages, max_tokens, temperature, stop, *, task_id, return_tokens):
        call = self.calls.popleft()
        assert call['kind'] == 'signature' and task_id == self.task_id
        assert max_tokens == call['max_new_tokens'] and len(messages) == len(call['requests'])
        return deepcopy(call['outputs'])


def replay(builder_type, task, revision, protocol):
    first = next(call for call in revision['calls'] if call['kind'] == 'finite')
    state = json.loads(first['requests'][0]['state'])
    settings = state['construction_limits']
    shared = yaml.safe_load(protocol['shared_config_text'])
    service = RecordedService(revision['calls'], task['task_id'])
    execution = protocol['method']['settings']['rollout']['execution']
    trace = {}
    instance = builder_type(task['trace']['query'], state['public_tools'], state['answer_schema'],
        service, task['task_id'], shared['generation'], settings,
        load_prompt(settings['prompts']), execution, trace, feedback=state['feedback'])
    declaration = instance.build_action()
    assert not service.calls
    outputs = [call for call in trace['calls'] if call['kind'] == 'compiled_signature']
    return declaration, outputs, compile_declaration(declaration, execution)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    config = json.loads((ROOT / args.config).read_text())
    assert (ROOT / config['reference_builder']).exists()
    accepted, conflicts, per_run = [], [], []
    started = time.perf_counter()
    for run in config['runs']:
        directory = ROOT / run
        protocol = json.loads((directory / 'protocol.json').read_text())
        counts = Counter()
        for source in sorted(directory.glob('task-*.json')):
            task = json.loads(source.read_text())
            for turn in task['trace']['rounds']:
                revision = turn.get('revision', {})
                if 'observation' not in revision:
                    continue
                observation = revision['observation']
                identity = {'source': str(source.relative_to(ROOT)), 'task_id': task['task_id'], 'turn': turn['turn']}
                if observation['accepted']:
                    old, old_outputs, old_compiled = replay(ReferenceBuilder, task, revision, protocol)
                    new, new_outputs, new_compiled = replay(DeclarationBuilder, task, revision, protocol)
                    actual_outputs = [call for call in revision['calls'] if call['kind'] == 'compiled_signature']
                    assert canonical(old) == canonical(revision['declaration']), identity
                    assert canonical(old_outputs) == canonical(actual_outputs), identity
                    assert canonical(old_compiled) == canonical(observation['declaration']), identity
                    assert canonical(new) == canonical(old), identity
                    assert canonical(new_outputs) == canonical(old_outputs), identity
                    assert canonical(new_compiled) == canonical(old_compiled), identity
                    accepted.append({**identity, 'fields': len(new['fields']), 'signature_calls': len(new_outputs),
                        'recorded_program_exact': True, 'new_program_exact': True,
                        'compiled_signature_outputs_exact': True, 'topological_compilation_exact': True})
                    counts['accepted_exact'] += 1
                elif observation['error'] == config['domain_conflict_error']:
                    try:
                        replay(ReferenceBuilder, task, revision, protocol)
                    except AssertionError as error:
                        assert str(error) == config['domain_conflict_error'], identity
                    else:
                        raise AssertionError({'unexpected_reference_acceptance': identity})
                    try:
                        new, outputs, compiled = replay(DeclarationBuilder, task, revision, protocol)
                    except AssertionError as error:
                        assert str(error) in config['remaining_errors'], {'source': identity, 'error': str(error)}
                        status = config['remaining_errors'][str(error)]
                        conflicts.append({**identity, 'status': status, 'error': str(error),
                                          'source_signatures': observation['source_signatures']})
                        counts[status] += 1
                    else:
                        conflicts.append({**identity, 'status': 'compiled', 'fields': len(new['fields']),
                                          'program': new, 'compiled_signature_outputs': outputs})
                        counts['previous_domain_conflicts_compiled'] += 1
        per_run.append({'run': run, 'counts': dict(counts)})
    report = {'stage': '055', 'config': args.config, 'reference_builder': config['reference_builder'],
        'scope': 'Replay recorded construction choices and full native signature outputs; no model inference. Exact type-preserving comparison of complete fields/action/answer, compiled signature outputs, named bindings and topologically compiled declaration.',
        'accepted_declarations_exact': len(accepted), 'domain_conflicts_replayed': len(conflicts),
        'domain_conflict_outcomes': dict(Counter(row['status'] for row in conflicts)),
        'elapsed_seconds': time.perf_counter() - started, 'runs': per_run,
        'accepted': accepted, 'domain_conflicts': conflicts,
        'limits': 'Exact accepted compiler output does not establish identical model choices under the changed lexical-binding prompt. Conflicting-role programs were previously rejected; their new acceptance is an explicit language-semantics change.'}
    (ROOT / config['output']).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: value for key, value in report.items() if key not in ('accepted', 'domain_conflicts')}))


if __name__ == '__main__':
    main()
