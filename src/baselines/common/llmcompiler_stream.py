import asyncio
from concurrent.futures import Future
from queue import Empty
import re
from types import FunctionType

from langchain.schema import Generation, LLMResult, OutputParserException

from baselines.common.llmcompiler import SchemaCompiler, instantiate_task, solve_with_models
from src.llm_compiler.output_parser import ACTION_PATTERN
from src.llm_compiler.planner import LLMCompilerCallback, Planner, StreamingGraphParser
from src.llm_compiler.task_fetching_unit import TaskFetchingUnit
from baselines.official_llmcompiler.adapter import CallbackLLM


class StreamingCallbackLLM(CallbackLLM):
    async def _agenerate(self, prompts, stop=None, run_manager=None, **kwargs):
        service = self.complete.func.__self__
        task_id = self.complete.keywords['task_id']

        async def generate(prompt):
            request, channel = service.open_stream([{'role': 'user', 'content': prompt}],
                self.max_tokens, self.temperature, stop, task_id=task_id)
            chunks, arrivals = [], []
            try:
                while True:
                    try:
                        item, emitted = await asyncio.to_thread(channel.get, True, service.deadlines.remaining(task_id))
                    except Empty as error:
                        raise TimeoutError('Streaming sample deadline exceeded: ' + task_id) from error
                    if isinstance(item, Future):
                        text = item.result()
                        completed = emitted
                        assert ''.join(chunks) == text, 'Incremental text differs from the completed model response.'
                        break
                    chunks.append(item)
                    arrivals.append(emitted)
                    for fragment in item.splitlines(keepends=True):
                        if run_manager is not None:
                            await run_manager.on_llm_new_token(fragment)
            finally:
                service.release_stream(request)
            self.trace.append({'type': 'generation', 'role': self.role, 'prompt': prompt,
                'text': text, 'max_tokens': self.max_tokens, 'temperature': self.temperature, 'stop': stop,
                'stream_chunk_arrivals': arrivals, 'stream_completed': completed})
            return text

        responses = await asyncio.gather(*(generate(prompt) for prompt in prompts))
        return LLMResult(generations=[[Generation(text=text)] for text in responses])


class SchemaStreamingGraphParser(StreamingGraphParser):
    def _match_buffer_and_generate_task(self, suffix):
        match = re.match(ACTION_PATTERN, self.buffer)
        if match is None:
            return super()._match_buffer_and_generate_task(suffix)
        index, name, arguments, _ = match.groups()
        task = instantiate_task(self.tools, int(index), name, arguments, self.thought)
        self.thought = ''
        return task


class SchemaStreamingPlanner(Planner):
    async def aplan(self, inputs, task_queue, is_replan, callbacks=None, **kwargs):
        callback = LLMCompilerCallback(queue=task_queue, tools=self.tools)
        callback.raise_error = True
        callback._parser = SchemaStreamingGraphParser(tools=self.tools)
        handlers = [callback, *(callbacks or [])]
        await self.run_llm(inputs, is_replan=is_replan, callbacks=handlers)


class DependencyCheckedTaskFetchingUnit(TaskFetchingUnit):
    def set_tasks(self, tasks):
        available = self.tasks.keys() | tasks.keys()
        missing = {identity: sorted(set(task.dependencies) - available)
                   for identity, task in tasks.items() if set(task.dependencies) - available}
        if missing:
            raise OutputParserException(str(missing))
        super().set_tasks(tasks)


class StreamingSchemaCompiler(SchemaCompiler):
    # Keep upstream orchestration bytecode and replace only its dependency admission boundary.
    _acall = FunctionType(SchemaCompiler._acall.__code__,
        {**SchemaCompiler._acall.__globals__, 'TaskFetchingUnit': DependencyCheckedTaskFetchingUnit},
        SchemaCompiler._acall.__name__, SchemaCompiler._acall.__defaults__, SchemaCompiler._acall.__closure__)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.planner.__class__ = SchemaStreamingPlanner


def solve(task, environment, complete, settings, prompts):
    return solve_with_models(task, environment, complete, settings, prompts,
                             StreamingCallbackLLM, settings['planner_stream'], StreamingSchemaCompiler)
