import asyncio
import json
from pathlib import Path

from langchain.schema import Generation, LLMResult

from baselines.common.llmcompiler import SchemaCompiler, solve_with_models
from baselines.common.resources import ADAPTER_SETTINGS
from baselines.official_llmcompiler.adapter import CallbackLLM


TEMPLATES = json.loads((Path(__file__).parent / 'template/prompts_v2.json').read_text())['llmcompiler']


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
    prompts = {**prompts, 'planner': prompts['planner'] + TEMPLATES['planner_syntax']}
    return solve_with_models(task, environment, complete, settings, prompts, RoleCallback,
                             ADAPTER_SETTINGS['llmcompiler']['planner_stream'], RoleCompiler)
