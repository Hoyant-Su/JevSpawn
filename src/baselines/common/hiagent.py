import json

from baselines.common.errors import InvalidOutputError, TaskLimitError
from baselines.common.react import EnvironmentAdapter
from baselines.hiagent.adapter import ModelTransport, original_agent
from baselines.common.resources import ADAPTER_SETTINGS


def solve(task, environment, complete, settings, prompts):
    model = ModelTransport(complete, settings)
    adapter = EnvironmentAdapter(environment, prompts)
    agent = original_agent(settings['source_directory'], environment={'EVALTASK': task['context']})(
        model, memory_size=settings['memory_size'], instruction=prompts['instruction'],
        examples=ADAPTER_SETTINGS['hiagent']['examples'], system_message=prompts['system'],
        need_goal=ADAPTER_SETTINGS['hiagent']['need_goal'],
        check_actions=prompts['check_actions'], use_parser=ADAPTER_SETTINGS['hiagent']['use_parser'])
    agent.reset(adapter.reset(), prompts['initial_observation'])
    for _ in range(settings['max_turns']):
        try:
            success, action = agent.run()
        except InvalidOutputError as error:
            error.trace = {'calls': model.calls, 'memory': agent.memory}
            raise
        if not success:
            raise RuntimeError('Original HiAgent failed to generate an action.')
        if action == prompts['check_actions']:
            observation, done = json.dumps(environment.display_tool_interface(True)), False
        else:
            observation, _, done, _ = adapter.step(action)
        agent.update(action, observation)
        if done:
            break
    if environment.answer is None:
        environment.deadline()
        raise TaskLimitError('HiAgent terminated within its declared limits without submitting an answer.')
    return {'task_id': task['task_id'], 'answer': environment.answer,
            'actions': environment.actions, 'memory': agent.memory, 'calls': model.calls,
            'subgoals': sum(item[0][0] == 'Subgoal' for item in agent.memory)}
