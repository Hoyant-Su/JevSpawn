import json
from pathlib import Path

from baselines.common.dyflow import parse_output, solve
from baselines.common.deadlines import SampleDeadlines
from baselines.common.scheduler import run_tasks
from baselines.common.errors import InvalidOutputError
from baselines.common.environment import TaskEnvironment
from jev_spawn.infra.prompts import load_prompt


FIXTURE = json.loads(Path('tests/fixtures/baselines/dyflow/common_interface.json').read_text())


class Environment:
    def __init__(self):
        self.answer, self.actions = None, []
        self.tools = FIXTURE['environment']['tools']
        self.tool_definitions = json.loads(Path(FIXTURE['environment']['definitions']).read_text())
        self.input_schemas = json.loads(Path(FIXTURE['environment']['input_schemas']).read_text())

    def deadline(self):
        return FIXTURE['environment']['deadline_seconds']

    def reset(self):
        return FIXTURE['environment']['reset']

    def execute(self, name, arguments):
        self.actions.append({'tool': name, 'arguments': arguments})
        if name == 'finish':
            self.answer = arguments
        return json.dumps(FIXTURE['environment']['observation']), name == 'finish'


def main():
    settings = json.loads(Path('configs/baselines/common/methods/dyflow.json').read_text())['settings']
    settings.update(FIXTURE['generation'])
    prompts = load_prompt('configs/baselines/common/schema/dyflow.json')
    designs = []

    def complete(messages, tokens, temperature):
        prompt = messages[0]['content']
        if prompt.startswith('\nYou are the workflow stage designer.'):
            index = len(designs)
            designs.append(index)
            instruction = ['SELF_CONSISTENCY_ENSEMBLE', 'ORGANIZE_SOLUTION'][index]
            params = {'instruction_type': instruction, 'input_keys': ['original_problem'],
                      'output_key': f'act_{index}', 'input_usage': 'solve'}
            if index:
                params['input_keys'].append('act_0')
            return [json.dumps({'stage_id': f'stage_{index + 1}', 'stage_description': 'test',
                    'operators': [{'operator_id': f'op_{index}', 'operator_description': 'test', 'params': params}]})]
        if prompt.startswith('You are an impartial adjudicator'):
            return [json.dumps(FIXTURE['responses']['selector'])]
        if prompt.startswith('You are summarizing'):
            return [FIXTURE['responses']['summary']]
        if prompt.startswith('Return ONLY'):
            return [json.dumps(FIXTURE['expected']['answer'])]
        return [FIXTURE['responses']['ensemble']]

    result = solve({'task_id': 'cpu-interface', 'answer_schema': {'type': 'object'}}, Environment(), complete, settings, prompts)
    assert result['answer'] == FIXTURE['expected']['answer']
    assert len(result['calls']) == FIXTURE['expected']['calls']
    assert len(result['design_history']) == FIXTURE['expected']['stages']
    assert len(result['state']['summarized_stages']) == FIXTURE['expected']['summaries']
    ensemble = result['state']['actions']['act_0']['ensemble_info']
    assert len(ensemble['all_solutions']) == FIXTURE['expected']['ensemble_samples'] and ensemble['selected_index'] == FIXTURE['expected']['selected_index']
    print('PASS: original workflow, five ensemble samples, selector, summary, submission, 10 calls.')

    designs.clear()

    def invalid_selector(messages, *args, **kwargs):
        if messages[0]['content'].startswith('You are an impartial adjudicator'):
            return ['{}']
        return complete(messages, *args, **kwargs)

    try:
        solve({'task_id': 'cpu-selector-failure'}, Environment(), invalid_selector, settings, prompts)
    except InvalidOutputError:
        print('PASS: missing selector choice fails without selecting a substitute candidate.')
    else:
        raise AssertionError('Invalid selector output was accepted.')

    invalid_cases = ['stage_json', 'stage_schema', 'tool_json', 'tool_schema', 'final_json', 'final_schema']

    def solve_invalid(task):
        case = task['task_id']

        def malformed(messages, *args, **kwargs):
            if messages[0]['content'].startswith('\nYou are the workflow stage designer.'):
                if case == 'stage_json':
                    return ['not JSON']
                if case == 'stage_schema':
                    return ['{}']
                instruction = 'TOOL_CALL' if case.startswith('tool') else 'ORGANIZE_SOLUTION'
                return [json.dumps({'stage_id': 'stage_1', 'stage_description': 'test', 'operators': [
                    {'operator_id': 'op_1', 'operator_description': 'test', 'params': {
                        'instruction_type': instruction, 'input_keys': ['original_problem'], 'output_key': 'act_0'}}]})]
            return ['{}' if case.endswith('schema') else 'not JSON']

        return solve(task, Environment(), malformed, settings, prompts)

    tasks = [{'task_id': case, 'answer_schema': {'type': 'object', 'required': ['answer']}}
             for case in invalid_cases]
    records = []
    results, _ = run_tasks(tasks, solve_invalid, FIXTURE['scheduler']['batch_size'], SampleDeadlines(FIXTURE['scheduler']['timeout_seconds']),
                           lambda index, record: records.append(record))
    assert len(records) == len(invalid_cases)
    assert all(record['status'] == 'invalid_output' and record['answer'] is None for record in results)
    print('PASS: malformed stage/tool/final JSON and schema preserve all six denominator records.')

    source_tasks = [json.loads(line) for line in Path('data/baselines/common/qualification/all_datasets_14.jsonl').read_text().splitlines()]
    ranking_task = next(task for task in source_tasks if task['kind'] == 'ranking')
    tool_settings = json.loads(Path('configs/baselines/common/environment.json').read_text())
    requested = [row['document_id'] for row in ranking_task['source']['candidates'][:tool_settings['documents_per_read'] + 1]]

    def overlimit(messages, *args, **kwargs):
        if messages[0]['content'].startswith('\nYou are the workflow stage designer.'):
            return [json.dumps({'stage_id': 'stage_1', 'stage_description': 'read documents', 'operators': [
                {'operator_id': 'op_1', 'operator_description': 'read', 'params': {
                    'instruction_type': 'TOOL_CALL', 'input_keys': ['original_problem'], 'output_key': 'act_0'}}]})]
        return [json.dumps({'tool': 'read_documents', 'arguments': {'ids': requested}})]

    actual_environment = TaskEnvironment(ranking_task, tool_settings, Path('logs/baselines/common/cpu-tools'),
                                         deadline=lambda: FIXTURE['environment']['deadline_seconds'])
    try:
        solve(ranking_task, actual_environment, overlimit, settings, prompts)
    except InvalidOutputError as error:
        assert 'tool arguments' in str(error)
        assert actual_environment.actions == [] and actual_environment.answer is None
    else:
        raise AssertionError('Too many documents were accepted.')
    print('PASS: actual shared document read limit becomes invalid output without an observation or answer.')

    class FailedToolEnvironment(TaskEnvironment):
        def execute(self, name, arguments):
            raise RuntimeError('unexpected tool infrastructure failure')

    requested.pop()
    failed_environment = FailedToolEnvironment(ranking_task, tool_settings, Path('logs/baselines/common/cpu-tools'),
                                               deadline=lambda: FIXTURE['environment']['deadline_seconds'])
    try:
        solve(ranking_task, failed_environment, overlimit, settings, prompts)
    except RuntimeError as error:
        assert str(error) == 'unexpected tool infrastructure failure'
    else:
        raise AssertionError('Unexpected tool failure was swallowed.')
    print('PASS: unrelated tool RuntimeError propagates.')

    def failed(*args, **kwargs):
        raise ValueError('required model failure')

    try:
        solve({'task_id': 'cpu-failure'}, Environment(), failed, settings, prompts)
    except (ValueError, RuntimeError) as error:
        assert 'required model failure' in str(error)
    else:
        raise AssertionError('Required failure was swallowed.')
    print('PASS: required model failure propagates.')

    batches = json.loads(Path(FIXTURE['trace']['batches']).read_text())
    replayed = []
    for batch in batches:
        for index, task_id in enumerate(batch['task_ids']):
            if (task_id in FIXTURE['trace']['task_ids'] and
                    batch['texts'][index].startswith(FIXTURE['trace']['fence_open']) and
                    batch['messages'][index][0]['content'].startswith(FIXTURE['trace']['tool_prompt_prefix'])):
                value = parse_output(batch['texts'][index], 'recorded tool call')
                assert value['tool'] in FIXTURE['trace']['expected_tools']
                replayed.append(task_id)
    assert set(replayed) == set(FIXTURE['trace']['task_ids'])
    for text in FIXTURE['interface']['malformed']:
        try:
            parse_output(text, 'CPU framing')
        except InvalidOutputError:
            pass
        else:
            raise AssertionError('Malformed framing was accepted.')
    seen = []

    def interface_complete(messages, *args, **kwargs):
        prompt = messages[0]['content']
        if prompt.startswith('\nYou are the workflow stage designer.'):
            instruction = FIXTURE['interface']['instructions'][len(seen)]
            return [json.dumps({'stage_id': instruction, 'stage_description': 'interface contract', 'operators': [
                {'operator_id': instruction, 'operator_description': 'interface contract', 'params': {
                    'instruction_type': instruction, 'input_keys': [], 'output_key': instruction}}]})]
        if prompt.startswith('You are summarizing'):
            return [FIXTURE['responses']['summary']]
        start = prompt.index('{"task_interface":')
        interface = json.JSONDecoder().raw_decode(prompt[start:])[0]['task_interface']
        seen.append(interface)
        response = FIXTURE['interface']['tool_call' if 'tools' in interface else 'final']
        return [FIXTURE['interface']['fence'].format(payload=json.dumps(response))]

    environment = Environment()
    result = solve(FIXTURE['interface']['task'], environment, interface_complete, settings, prompts)
    assert seen[0]['tools']['calculate']['input_schema'] == environment.input_schemas['calculate']
    assert seen[1]['answer_schema'] == FIXTURE['interface']['task']['answer_schema']
    assert result['answer'] == FIXTURE['interface']['final']
    print('PASS: real failed tool fences parse unchanged; empty model input_keys retain immutable tool/answer contracts.')


if __name__ == '__main__':
    main()
