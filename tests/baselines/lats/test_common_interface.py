from copy import deepcopy
import json
from pathlib import Path

from baselines.common.deadlines import SampleDeadlines
from baselines.common.errors import InvalidOutputError
from baselines.common.scheduler import run_tasks
from baselines.tool_agents.tools import calculate
from baselines.common.lats import solve
from jev_spawn.infra.prompts import load_prompt


FIXTURE = json.loads(Path('tests/fixtures/baselines/lats/common_interface.json').read_text())


class Environment:
    def __init__(self):
        self.answer, self.actions = None, []
        self.done, self.tool_timings = False, []
        self.tools = FIXTURE['environment']['tools']
        self.input_schemas = json.loads(Path('configs/baselines/common/schema/tool_inputs.json').read_text())
        self.input_schemas['finish'] = FIXTURE['environment']['finish_schema']

    def deadline(self):
        return FIXTURE['environment']['deadline_seconds']

    def fork(self):
        return deepcopy(self)

    def reset(self):
        return FIXTURE['environment']['reset']

    def execute(self, name, arguments):
        self.actions.append({'tool': name, 'arguments': arguments})
        if name == 'finish':
            self.answer = arguments
            self.done = True
        return json.dumps(FIXTURE['environment']['observation']), name == 'finish'


class CalculatorEnvironment(Environment):
    def execute(self, name, arguments):
        if name == 'calculate':
            return json.dumps(calculate(arguments['expression'], FIXTURE['environment']['calculator'])), False
        return super().execute(name, arguments)


def main():
    settings = json.loads(Path('configs/baselines/common/methods/lats.json').read_text())['settings']
    settings.update(FIXTURE['settings_override'])
    prompts = load_prompt('configs/baselines/common/schema/lats.json')
    batches = []
    answer = FIXTURE['answer']

    def complete(messages, tokens, temperature, n=FIXTURE['completion_batch_default'], stop=None):
        prompt = messages[0]['content']
        batches.append(n)
        if messages[0]['role'] == 'system':
            return [FIXTURE['value_template'].format(score=FIXTURE['score'])]
        if prompt.startswith(FIXTURE['reflection_prefix']):
            return [FIXTURE['reflection']]
        if FIXTURE['first_observation'] not in prompt:
            return [FIXTURE['calculate_template'].format(index=index)
                    for index in range(n)]
        return [FIXTURE['submit_template'].format(index=index, answer=json.dumps(answer)) for index in range(n)]

    result = solve({'task_id': FIXTURE['task_ids']['branching']}, Environment(), complete, settings, prompts)
    assert result['answer'] == answer
    assert len(result['terminal_estimates']) >= FIXTURE['expected']['minimum_terminal_estimates']
    assert all(record['value'] == FIXTURE['expected']['terminal_value'] for record in result['terminal_estimates'])
    assert any(node['visits'] >= FIXTURE['expected']['minimum_visits'] for node in result['nodes'])
    assert result['reflections']
    assert batches.count(FIXTURE['expected']['batch_size']) >= FIXTURE['expected']['minimum_branch_batches']
    print('PASS: original MCTS branching, LM scoring, rollout, backpropagation, reflection, complete JSON arrays.')

    for score in FIXTURE['terminal_scores']:
        def terminal(messages, tokens, temperature, n=FIXTURE['completion_batch_default'], stop=None):
            if messages[0]['role'] == 'system':
                return [FIXTURE['value_template'].format(score=score)]
            return [FIXTURE['terminal_template'].format(index=index, answer=json.dumps(answer)) for index in range(n)]

        terminal_result = solve({'task_id': FIXTURE['task_ids']['terminal_template'].format(score=score)}, Environment(), terminal, settings, prompts)
        assert terminal_result['answer'] == answer
        assert terminal_result['terminal_value'] == score / FIXTURE['score_scale']
    print('PASS: all-terminal expansion accepts score 10 and safely exhausts score 9.')

    def invalid(messages, *args, **kwargs):
        if messages[0]['role'] == 'system':
            return [FIXTURE['invalid_value']]
        return complete(messages, *args, **kwargs)

    try:
        solve({'task_id': FIXTURE['task_ids']['invalid_value']}, Environment(), invalid, settings, prompts)
    except InvalidOutputError:
        print('PASS: missing value estimate terminates without fabricated reward.')
    else:
        raise AssertionError('Invalid value estimate was accepted.')

    tasks = [{'task_id': name} for name in FIXTURE['rejected_actions']]

    def invalid_tool(task):
        output = FIXTURE['rejected_actions'][task['task_id']]

        def complete_tool(messages, tokens, temperature, n=FIXTURE['completion_batch_default'], stop=None):
            prompt = messages[0]['content']
            if messages[0]['role'] == 'system':
                return [FIXTURE['value_template'].format(score=FIXTURE['score_scale'])]
            if FIXTURE['first_observation'] in prompt:
                assert 'error' in prompt
                return [FIXTURE['correction_template'].format(answer=json.dumps(answer))] * n
            return [FIXTURE['rejected_template'].format(output=output)] * n

        return solve(task, CalculatorEnvironment(), complete_tool, settings, prompts)

    committed = []
    results, _ = run_tasks(tasks, invalid_tool, FIXTURE['scheduler_workers'], SampleDeadlines(FIXTURE['environment']['deadline_seconds']),
                           lambda index, result: committed.append(result))
    assert len(committed) == len(tasks)
    assert all(result['status'] == 'completed' and result['answer'] == answer for result in results)
    assert all(any('error' in action for action in result['actions']) for result in results)
    print('PASS: rejected calculator, JSON and bracket actions reach the policy as real errors before recovery.')

    class FailedEnvironment(Environment):
        def execute(self, name, arguments):
            raise RuntimeError(FIXTURE['runtime_error'])

    try:
        solve({'task_id': FIXTURE['task_ids']['runtime_failure']}, FailedEnvironment(), complete, settings, prompts)
    except RuntimeError as error:
        assert str(error) == FIXTURE['runtime_error']
    else:
        raise AssertionError('Unexpected infrastructure failure was swallowed.')
    print('PASS: unexpected tool infrastructure failures propagate.')


if __name__ == '__main__':
    main()
