import argparse
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from baselines.common.config import SharedConfig
from baselines.common.errors import TaskLimitError
from baselines.common.transition_budget import TransitionBudget
from baselines.common.llmcompiler_v4 import SupervisedTaskFetchingUnit
from src.llm_compiler.task_fetching_unit import Task


async def run(settings):
    shared = SharedConfig.load(settings['shared_config'])
    execute = Mock()
    environment = TransitionBudget(SimpleNamespace(execute=execute), shared.runtime.max_turns,
        settings['submission_tool'], shared.runtime.max_turns, [])

    async def invoke():
        return environment.execute(settings['action'], {})

    scheduler = SupervisedTaskFetchingUnit()
    scheduler.set_tasks({identity: Task(identity, settings['action'], invoke, [], [])
                         for identity in settings['task_identities']})
    try:
        await asyncio.wait_for(scheduler.schedule(), settings['timeout_seconds'])
    except TaskLimitError as error:
        execute.assert_not_called()
        assert all(task.observation is None for task in scheduler.tasks.values())
        result = {'status': 'passed', 'exception': type(error).__name__,
            'message': str(error), 'environment_called': False,
            'observation_fabricated': False, 'shared_max_turns': shared.runtime.max_turns,
            'simultaneous_workers': len(scheduler.tasks),
            'scope': 'CPU propagation regression at the actual shared transition limit; no inference or task score.'}
        Path(settings['output']).write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result))
    else:
        raise AssertionError('The exhausted transition budget must propagate to the sample runner.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    asyncio.run(run(json.loads(parser.parse_args().config.read_text())))
