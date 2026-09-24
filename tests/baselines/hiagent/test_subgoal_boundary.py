import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from baselines.common.errors import InvalidOutputError
from baselines.hiagent.adapter import original_agent


def test_actual_recursive_retrieval_outputs_fail_as_model_output():
    batches = json.loads(Path('runs/native-baselines-formal-20260923/hiagent/autoplanbench/'
                              'session-0000/batches.json').read_text())
    task_id = 'autoplanbench_grippers_instance_11'
    outputs = [output for batch in batches
               for identity, output in zip(batch['task_ids'], batch['output_texts'], strict=True)
               if identity == task_id]
    replies = iter(outputs[-2:])
    model = SimpleNamespace(engine='test', context_length=16384, max_tokens=2048,
                            num_tokens_from_messages=lambda messages: 0,
                            generate=lambda system, prompt: (True, next(replies)))
    core = original_agent('external/HiAgent/agentboard/agents', environment={'EVALTASK': 'grippers'})
    agent = core(model, instruction='', need_goal=True)
    agent.reset('Task context', 'Initial observation')
    agent.run()
    assert [entry[0][0] for entry in agent.memory] == ['Observation', 'Subgoal', 'Subgoal']
    with pytest.raises(InvalidOutputError, match='no intervening action or observation'):
        agent.make_prompt()


def test_valid_action_observation_history_preserved():
    model = SimpleNamespace(engine='test', context_length=16384, max_tokens=2048,
                            num_tokens_from_messages=lambda messages: 0)
    core = original_agent('external/HiAgent/agentboard/agents', environment={'EVALTASK': 'grippers'})
    agent = core(model)
    agent.reset('Task context', 'Initial observation')
    agent.memory.extend([[('Subgoal', 'First')], [('Action', 'Executed'), ('Observation', 'Observed')],
                         [('Subgoal', 'Second')]])
    prompt = agent.make_prompt()
    assert '1 Subgoal: First\nObservation: Observed\n2 Subgoal: Second' in prompt
