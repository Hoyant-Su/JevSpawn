from copy import deepcopy
import json
from pathlib import Path
import statistics

from transformers import AutoTokenizer
import yaml

from baselines.common.environment import TaskEnvironment
from baselines.common.schema_environment import SchemaObservationEnvironment
from methods.evidence_flow.environment import EvidenceEnvironment
from jev_spawn.infra.prompts import load_prompt


FIXTURE = json.loads(Path('tests/baselines/common/fixtures/search_schema_v1.json').read_text())


def main():
    protocol = json.loads(Path(FIXTURE['protocol']).read_text())
    source = Path(FIXTURE['failures']).parent
    batches = json.loads((source / 'batches.json').read_text())
    failures = json.loads(Path(FIXTURE['failures']).read_text())
    schema = load_prompt(FIXTURE['schema'])
    model = yaml.safe_load(protocol['shared_config_text'])['model']
    tokenizer = AutoTokenizer.from_pretrained(model['path'], local_files_only=True)
    corpus = EvidenceEnvironment(**protocol['tools']['evidence'])
    first, last = {}, {}
    for batch in batches:
        for index, identity in enumerate(batch['task_ids']):
            first.setdefault(identity, (batch['messages'][index], batch['role_messages'][index]))
            last[identity] = batch['messages'][index]
    for failure in failures:
        last[failure['task_id']] = failure['messages']

    def size(messages):
        rendered = tokenizer.apply_chat_template(messages, tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
        return len(tokenizer(rendered, add_special_tokens=False)['input_ids'])

    rows = []
    for task in protocol['tasks']:
        options = dict(deadline=lambda: None, evidence=corpus)
        original = TaskEnvironment(task, protocol['tools'], FIXTURE['tool_directory'], **options)
        candidate = SchemaObservationEnvironment(task, protocol['tools'], FIXTURE['tool_directory'],
                                                read_id_schema=schema, **options)
        old = last[task['task_id']]
        new = deepcopy(old)
        assert old[1]['content'] == original.reset()
        new[1]['content'] = candidate.reset()
        initial, roles = first[task['task_id']]
        initial_question = '\n\n'.join(m['role'] + ':\n' + m['content'] for m in initial)
        lengths = []
        for history in (old, new):
            question = '\n\n'.join(m['role'] + ':\n' + m['content'] for m in history)
            expanded = [[{**m, 'content': m['content'].replace(initial_question, question)}
                         for m in role] for role in roles]
            lengths.append([size(role) for role in expanded])
        rows.append({'task_id': task['task_id'], 'canonical_original': size(old),
            'canonical_candidate': size(new), 'original_roles': lengths[0],
            'candidate_roles': lengths[1], 'canonical_added': size(new) - size(old),
            'role_history_added': sum(lengths[1]) - sum(lengths[0])})
    by_id = {row['task_id']: row for row in rows}
    assert all(by_id[failure['task_id']]['original_roles'] == failure['role_input_tokens']
               for failure in failures)
    report = {'stage': 'Isolated static semantic read-ID schema CPU qualification; replaces rejected envelope candidate, not enabled in active runs',
        'source_protocol': FIXTURE['protocol'], 'model_tokenizer': model['path'],
        'scope': 'Actual last request of each original task, including rejected requests; observed actions held fixed. Only the initial model-visible read ID item schema changes. Original tool observations are byte-identical; runtime enforces the exact observed-ID union.',
        'cpu_tests_passed': ['actual_results_and_cumulative_read_schema',
                            'concurrent_search_preserves_results_and_disclosed_ids',
                            'unobserved_read_is_still_rejected'],
        'task_count': len(rows), 'rows': rows}
    for name in ('canonical_added', 'role_history_added'):
        values = [row[name] for row in rows]
        report[name] = {'min': min(values), 'median': statistics.median(values), 'max': max(values)}
    destination = Path('results/baselines/common/monitoring/search_schema_v1_qualification_002.json')
    destination.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({name: report[name] for name in ('task_count', 'canonical_added', 'role_history_added')}))


if __name__ == '__main__':
    main()
