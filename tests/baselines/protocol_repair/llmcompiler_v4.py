import asyncio

from src.llm_compiler import llm_compiler
from src.llm_compiler.task_fetching_unit import TaskFetchingUnit

from llmcompiler_v3 import solve


class SupervisedTaskFetchingUnit(TaskFetchingUnit):
    async def schedule(self):
        workers = set()
        try:
            while not self._all_tasks_done():
                for identity in self._get_all_executable_tasks():
                    workers.add(asyncio.create_task(self._run_task(self.tasks[identity])))
                    self.remaining_tasks.remove(identity)
                completed, _ = await asyncio.wait(workers, return_when=asyncio.FIRST_COMPLETED)
                for worker in completed:
                    worker.result()
                workers.difference_update(completed)
        finally:
            for worker in workers:
                worker.cancel()
            await asyncio.gather(*workers, return_exceptions=True)


llm_compiler.TaskFetchingUnit = SupervisedTaskFetchingUnit
