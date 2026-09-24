import asyncio
import json
from pathlib import Path

from langchain.schema import Generation, LLMResult

from src.llm_compiler import llm_compiler
from src.llm_compiler.task_fetching_unit import TaskFetchingUnit

from baselines.common.llmcompiler import SchemaCompiler, solve_with_models
from baselines.common.resources import ADAPTER_SETTINGS
from baselines.official_llmcompiler.adapter import CallbackLLM


TEMPLATES = json.loads(Path('configs/baselines/common/templates/llmcompiler_v4.json').read_text())


class RoleCallback(CallbackLLM):
    async def _agenerate(self, prompts, stop=None, run_manager=None, **kwargs):
        messages = [[{'role': 'system', 'content': TEMPLATES['roles'][self.role]},
                     {'role': 'user', 'content': prompt}] for prompt in prompts]
        responses = await asyncio.gather(*[
            asyncio.to_thread(self.complete, request, self.max_tokens, self.temperature, stop=stop)
            for request in messages])
        for prompt, request, response in zip(prompts, messages, responses):
            self.trace.append({'type': 'generation', 'role': self.role, 'prompt': prompt,
                               'messages': request, 'text': response[0], 'max_tokens': self.max_tokens,
                               'temperature': self.temperature, 'stop': stop})
        return LLMResult(generations=[[Generation(text=response[0])] for response in responses])


class RoleCompiler(SchemaCompiler):
    def __init__(self, **kwargs):
        agent = kwargs['agent_llm']
        kwargs['agent_llm'] = RoleCallback(complete=agent.complete, role=agent.role,
                                         max_tokens=agent.max_tokens, temperature=agent.temperature,
                                         trace=agent.trace)
        super().__init__(**kwargs)

    async def join(self, input_query, agent_scratchpad, is_final):
        instruction = self.joinner_prompt_final if is_final else self.joinner_prompt
        prompt = TEMPLATES['joiner_input'].format(task=input_query, observations=agent_scratchpad,
                                                 instruction=instruction)
        response = await self.agent.arun(prompt, callbacks=[self.executor_callback] if self.benchmark else None)
        thought, answer, is_replan = self._parse_joinner_output(response)
        return thought, answer, False if is_final else is_replan


def solve(task, environment, complete, settings, prompts):
    settings = {**settings, 'planning_rounds': settings['max_turns']}
    prompts = {**prompts, 'planner': prompts['planner'] + TEMPLATES['planner_syntax']}
    return solve_with_models(task, environment, complete, settings, prompts, RoleCallback,
                             ADAPTER_SETTINGS['llmcompiler']['planner_stream'], RoleCompiler)


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
