import ast
import asyncio
from functools import partial
from io import StringIO
import json
import re
import tokenize

import jsonschema

from langchain.schema import OutputParserException

from src.llm_compiler import output_parser, planner as planner_module
from src.llm_compiler.llm_compiler import LLMCompiler
from src.tools.base import Tool
from baselines.common.errors import InvalidOutputError
from baselines.common.tasks import render
from baselines.official_llmcompiler.adapter import CallbackLLM
from baselines.official_llmcompiler.planner_compat import support_empty_tools
from baselines.common.resources import ADAPTER_SETTINGS, TEMPLATES
from baselines.tool_agents.tools import ActionError


planner_module.generate_llm_compiler_prompt = support_empty_tools(planner_module.generate_llm_compiler_prompt)
ARGUMENT_GRAMMAR = ADAPTER_SETTINGS['llmcompiler']['argument_grammar']
IGNORED_TOKENS = {getattr(tokenize, name) for name in ARGUMENT_GRAMMAR['ignored_tokens']}


def positional_arguments(arguments):
    try:
        return ast.literal_eval('(' + arguments + ',)') if arguments.strip() else ()
    except (SyntaxError, ValueError) as error:
        raise OutputParserException(TEMPLATES['compiler_argument_error']) from error


output_parser._parse_llm_compiler_action_args = positional_arguments
original_instantiate = output_parser.instantiate_task


def argument_segments(source):
    segments, current, stack = [], [], []
    brackets = ARGUMENT_GRAMMAR['open_brackets']
    for token in tokenize.generate_tokens(StringIO(source).readline):
        if token.type in IGNORED_TOKENS:
            continue
        if token.type == tokenize.OP:
            if token.string in brackets:
                stack.append(brackets[token.string])
            elif token.string in brackets.values():
                if not stack or stack.pop() != token.string:
                    raise ValueError(TEMPLATES['compiler_argument_error'])
            elif token.string == ARGUMENT_GRAMMAR['comma'] and not stack:
                if not current:
                    raise ValueError(TEMPLATES['compiler_argument_error'])
                segments.append(current)
                current = []
                continue
        current.append(token)
    if current:
        segments.append(current)
    return segments


def literal_bindings(source, fields):
    values, keywords = [], {}
    for segment in argument_segments(source):
        first, *remaining = segment
        named = (first.type == tokenize.NAME and remaining and remaining[0].type == tokenize.OP
                 and remaining[0].string in ARGUMENT_GRAMMAR['named_separators'])
        if named:
            separator, *literal = remaining
            if first.string in keywords:
                raise ValueError(TEMPLATES['compiler_signature_error'])
            keywords[first.string] = ast.literal_eval(tokenize.untokenize(
                [(token.type, token.string) for token in literal]))
        else:
            if keywords:
                raise ValueError(TEMPLATES['compiler_signature_error'])
            values.append(ast.literal_eval(tokenize.untokenize(
                [(token.type, token.string) for token in segment])))
    if len(values) > len(fields) or set(keywords) != set(fields[len(values):]):
        raise ValueError(TEMPLATES['compiler_signature_error'])
    values.extend(keywords[field] for field in fields[len(values):])
    return values


def instantiate_task(tools, idx, tool_name, args, thought):
    fields = ([] if tool_name == ARGUMENT_GRAMMAR['join_tool'] else
              output_parser._find_tool(tool_name, tools).func.keywords['fields'])
    try:
        values = literal_bindings(args, fields)
    except (SyntaxError, ValueError, tokenize.TokenError) as error:
        raise OutputParserException(str(error)) from error
    return original_instantiate(tools, idx, tool_name, ', '.join(map(repr, values)), thought)


output_parser.instantiate_task = instantiate_task


class SchemaPlanParser(output_parser.LLMCompilerPlanParser):
    def parse(self, text):
        tasks = super().parse(text)
        missing = {identity: sorted(set(task.dependencies) - tasks.keys())
                   for identity, task in tasks.items() if set(task.dependencies) - tasks.keys()}
        if missing:
            raise OutputParserException(str(missing))
        return tasks


class SchemaCompiler(LLMCompiler):
    def _parse_joinner_output(self, raw_answer):
        thought, _, _ = super()._parse_joinner_output(raw_answer)
        actions = re.findall(TEMPLATES['compiler_joiner_action_pattern'], raw_answer, re.MULTILINE)
        if not actions:
            raise InvalidOutputError('Joiner did not produce a complete Finish or Replan action.')
        action, answer = actions[-1]
        return thought, answer, action == 'Replan'


async def invoke(*values, environment, name, fields):
    if len(values) != len(fields):
        error = ActionError(TEMPLATES['compiler_expected_arguments'].format(fields=', '.join(fields)))
        return json.dumps(environment.reject(name, values, error))
    arguments = dict(zip(fields, values, strict=True))
    observation, _ = await asyncio.to_thread(environment.execute, name, arguments)
    if name == 'calculate':
        result = json.loads(observation)
        return observation if 'error' in result else str(result['value'])
    return observation


def solve(task, environment, complete, settings, prompts):
    return solve_with_models(task, environment, complete, settings, prompts, CallbackLLM,
                             ADAPTER_SETTINGS['llmcompiler']['planner_stream'], SchemaCompiler)


def solve_with_models(task, environment, complete, settings, prompts, planner_type, planner_stream, compiler_type):
    trace, tools = [], []
    for name, definition in environment.tool_definitions.items():
        if name == 'finish':
            continue
        fields = list(definition['arguments'])
        signature = ', '.join(f'{field}: {definition["arguments"][field]}' for field in fields)
        tools.append(Tool(name=name,
            func=partial(invoke, environment=environment, name=name, fields=fields),
            description=TEMPLATES['compiler_tool_description'].format(
                name=name, signature=signature, description=definition['description'])))
    planner = planner_type(complete=complete, role='planner', trace=trace,
                          max_tokens=settings['max_new_tokens'], temperature=settings['temperature'])
    joiner = CallbackLLM(complete=complete, role='joiner', trace=trace,
                         max_tokens=settings['max_new_tokens'], temperature=settings['temperature'])
    compiler = compiler_type(tools=tools, planner_llm=planner, agent_llm=joiner,
        planner_example_prompt=prompts['planner'], planner_example_prompt_replan=prompts['planner'],
        planner_stop=settings['planner_stop'], planner_stream=planner_stream,
        joinner_prompt=prompts['joiner'], joinner_prompt_final=prompts['joiner_final'],
        max_replans=settings['planning_rounds'], benchmark=ADAPTER_SETTINGS['llmcompiler']['benchmark'])
    compiler.planner.output_parser = SchemaPlanParser(tools=tools)

    async def execute():
        question = environment.context(False, {})
        return await asyncio.wait_for(compiler.arun(question), timeout=environment.deadline())

    try:
        raw = asyncio.run(execute())
        answer = json.loads(raw)
        observation, done = environment.execute('finish', answer)
        if not done:
            raise InvalidOutputError(observation)
    except (OutputParserException, json.JSONDecodeError, jsonschema.ValidationError) as error:
        failure = InvalidOutputError(str(error))
        failure.trace = {'calls': trace, 'actions': environment.actions}
        raise failure from error
    except InvalidOutputError as error:
        error.trace = {'calls': trace, 'actions': environment.actions}
        raise
    return {'task_id': task['task_id'], 'answer': answer, 'calls': trace, 'actions': environment.actions}
