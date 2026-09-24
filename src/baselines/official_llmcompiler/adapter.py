import asyncio
import json
from functools import partial
from pathlib import Path
from typing import Any

from langchain.llms.base import BaseLLM
from langchain.schema import Generation, LLMResult

from src.llm_compiler.llm_compiler import LLMCompiler
from src.tools.base import Tool
from baselines.tool_agents.tools import calculate

from project_paths import ROOT
from jev_spawn.infra.prompts import load_prompt


PROMPT_FRAGMENTS = load_prompt('src/baselines/official_llmcompiler/adapter.py')


class CallbackLLM(BaseLLM):
    complete: Any
    role: str
    max_tokens: int
    temperature: float
    trace: Any

    @property
    def _llm_type(self):
        return 'shared-native-callback'

    def _generate(self, prompts, stop=None, run_manager=None, **kwargs):
        raise NotImplementedError('The official LLMCompiler adapter uses asynchronous inference.')

    async def _agenerate(self, prompts, stop=None, run_manager=None, **kwargs):
        responses = await asyncio.gather(*[
            asyncio.to_thread(self.complete, [{'role': 'user', 'content': prompt}],
                              self.max_tokens, self.temperature, n=1, stop=stop)
            for prompt in prompts])
        assert all(len(response) == 1 for response in responses)
        for prompt, response in zip(prompts, responses):
            self.trace.append({'type': 'generation', 'role': self.role, 'prompt': prompt,
                               'text': response[0], 'max_tokens': self.max_tokens,
                               'temperature': self.temperature, 'stop': stop})
        return LLMResult(generations=[[Generation(text=response[0])] for response in responses])


async def calculator(expression, limits, trace):
    try:
        result = str(calculate(expression, limits)['value'])
    except (ValueError, TypeError, SyntaxError, ZeroDivisionError, OverflowError) as error:
        result = PROMPT_FRAGMENTS['tool_error'].format(error_type=type(error).__name__, error=error)
    trace.append({'type': 'tool', 'tool': 'calculator', 'expression': expression, 'observation': result})
    return result


def make_compiler(complete, config, trace):
    schema = load_prompt(Path('configs/baselines/official_llmcompiler/schema') / config['prompt_schema'])
    tools = [Tool(name='calculator', func=partial(calculator, limits=config['calculator'], trace=trace),
                  description=schema['calculator'], stringify_rule=lambda args: PROMPT_FRAGMENTS['stringify_tool'].format(expression=args[0]))]
    planner = CallbackLLM(complete=complete, role='planner', max_tokens=config['planner_max_tokens'],
                          temperature=config['temperature'], trace=trace)
    joiner = CallbackLLM(complete=complete, role='joiner', max_tokens=config['joiner_max_tokens'],
                         temperature=config['temperature'], trace=trace)
    return LLMCompiler(tools=tools, planner_llm=planner,
                       planner_example_prompt=schema['planner_examples'],
                       planner_example_prompt_replan=schema['planner_examples'],
                       planner_stop=config['planner_stop'], planner_stream=False,
                       agent_llm=joiner, joinner_prompt=schema['joiner'],
                       joinner_prompt_final=schema['joiner_final'],
                       max_replans=config['max_plans'], benchmark=False)


async def solve(task, complete, config):
    schema = load_prompt(Path('configs/baselines/official_llmcompiler/schema') / config['prompt_schema'])
    trace = []
    engine = make_compiler(complete, config, trace)
    query = schema['question'].format(**task)
    answer = await asyncio.wait_for(engine.arun(query), timeout=config['task_timeout_seconds'])
    return {'task_id': task['task_id'], 'answer': answer, 'trace': trace}
