import ast
import json
from pathlib import Path
from threading import Lock

import jsonschema

from environments.arguments import validate_arguments

from baselines.common.errors import InvalidOutputError, TaskLimitError
from baselines.dyflow.adapter import SourceInterface, validate_selector as original_validate_selector
from baselines.tool_agents.tools import ActionError
from baselines.common.resources import ADAPTER_SETTINGS
from jev_spawn.infra.configuration import CORE


def parse_output(text, boundary):
    payload = text.strip()
    framing = CORE['framing']
    if payload.startswith(framing['json_open']):
        lines = payload.splitlines()
        if (len(lines) < framing['minimum_lines'] or lines[0] != framing['json_open'] or
                lines[-1] != framing['close']):
            raise InvalidOutputError(f'DyFlow {boundary} has an incomplete JSON code fence.')
        payload = '\n'.join(lines[1:-1])
    try:
        return json.loads(payload)
    except json.JSONDecodeError as error:
        raise InvalidOutputError(f'DyFlow {boundary} is not valid JSON: {error}') from error


def validate_output(value, schema, boundary):
    try:
        jsonschema.validate(value, schema)
    except jsonschema.ValidationError as error:
        raise InvalidOutputError(f'DyFlow {boundary} does not match its schema: {error.message}') from error
    return value


def validate_model_arguments(arguments, schema, boundary, trace):
    try:
        validate_arguments(arguments, schema)
    except json.JSONDecodeError as error:
        failure = InvalidOutputError(f'DyFlow {boundary} contains an invalid JSON scalar: {error}')
        failure.trace = trace
        raise failure from error


def validate_selector(text, count):
    try:
        original_validate_selector(text, count)
    except (ValueError, KeyError, TypeError) as error:
        raise InvalidOutputError(f'DyFlow ensemble selector is invalid: {error}') from error


class SourceTransport(SourceInterface):
    def visit_FunctionDef(self, node):
        node = self.generic_visit(node)
        if node.name == 'execute':
            for statement in ast.walk(node):
                if (isinstance(statement, ast.Raise) and isinstance(statement.exc, ast.Call)
                        and isinstance(statement.exc.func, ast.Name) and statement.exc.func.id == 'ValueError'
                        and any(isinstance(part, ast.Name) and part.id == 'full_key_path'
                                for argument in statement.exc.args for part in ast.walk(argument))):
                    statement.exc.func.id = 'ActionError'
                if isinstance(statement, ast.Try):
                    for handler in statement.handlers:
                        if any(isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                               and call.func.attr == 'add_error' for call in ast.walk(handler)):
                            handler.body.extend(ast.parse(
                                f'if not isinstance({handler.name}, (InvalidOutputError, ActionError)):\n'
                                '    raise').body)
                    if statement.finalbody:
                        for child in ast.walk(statement):
                            for field, value in ast.iter_fields(child):
                                if isinstance(value, list) and child in statement.finalbody:
                                    setattr(child, field, [ast.Pass() if isinstance(item, ast.Return) else item
                                                          for item in value])
            if any(isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name)
                   and item.target.id == 'status' for item in node.body):
                node.body.append(ast.Return(ast.Name(id='status', ctx=ast.Load())))
        if node.name == '_design_next_stage':
            for statement in ast.walk(node):
                if isinstance(statement, ast.Try):
                    statement.handlers.insert(0, ast.ExceptHandler(
                        type=ast.Name(id='InvalidOutputError', ctx=ast.Load()), name=None, body=[ast.Raise()]))
        if node.name == '_execute_code':
            node.body = ast.parse("raise RuntimeError('Code execution requires the shared task sandbox.')").body
        return node

    def visit_Assign(self, node):
        if any(isinstance(target, ast.Name) and target.id == 'max_iterations' for target in node.targets):
            node.value = ast.Name(id='MAX_STAGES', ctx=ast.Load())
        return super().visit_Assign(node)


def load_core(settings, service_factory):
    namespace = {'ModelService': service_factory, 'ExecutorLLMClient': object,
                 'validate_selector': validate_selector, 'MAX_STAGES': settings['workflow_stages'],
                 'InvalidOutputError': InvalidOutputError, 'ActionError': ActionError}
    for name in ['state.py', 'operator.py', 'workflow.py']:
        path = Path(settings['source_directory']) / 'dyflow/core' / name
        tree = ast.fix_missing_locations(SourceTransport().visit(ast.parse(path.read_text())))
        exec(compile(tree, str(path), 'exec'), namespace)
    return namespace


def solve(task, environment, complete, settings, prompts):
    calls, lock = [], Lock()

    class ModelService:
        def __init__(self, model=ADAPTER_SETTINGS['dyflow']['provider_model_label'], temperature=None,
                     role=ADAPTER_SETTINGS['dyflow']['provider_role']):
            self.model, self.role = model, role

        def generate(self, prompt, temperature=None, max_tokens=None):
            environment.deadline()
            with lock:
                record = {'call_id': len(calls), 'role': self.role, 'prompt': prompt,
                          'requested_temperature': temperature, 'requested_max_tokens': max_tokens,
                          'temperature': settings['temperature'], 'max_tokens': settings['max_new_tokens']}
                calls.append(record)
            output = complete([{'role': 'user', 'content': prompt}], settings['max_new_tokens'],
                              settings['temperature'])[0]
            record['output'] = output
            return {'response': output}

    core = load_core(settings, ModelService)
    core['DESIGN_STAGE_PROMPT'] += '\n' + prompts['designer']
    core['PROMPT_TEMPLATES']['TOOL_CALL'] = prompts['tool']
    core['PROMPT_TEMPLATES']['ORGANIZE_SOLUTION'] = prompts['organize']
    original = core['InstructExecutorOperator']

    class Operator(original):
        def execute(self, state, params):
            environment.deadline()
            state.original_problem = environment.reset()
            instruction = params['instruction_type']
            self.instruction_type = instruction
            if instruction not in settings['allowed_operators']:
                raise InvalidOutputError('Unsupported operator: ' + instruction)
            if instruction == 'SELF_CONSISTENCY_ENSEMBLE':
                params.update(num_samples=settings['ensemble_samples'],
                              ensemble_temperature=settings['temperature'], max_tokens=settings['max_new_tokens'])
            return super().execute(state, params)

        def _process_output(self, output, instruction):
            if instruction == 'TOOL_CALL':
                try:
                    call = validate_output(parse_output(output, 'tool call'), prompts['tool_schema'], 'tool call')
                except InvalidOutputError as error:
                    raise ActionError(str(error)) from error
                if call['tool'] == 'finish':
                    raise ActionError('DyFlow submits through ORGANIZE_SOLUTION; TOOL_CALL executes task actions.')
                observation, _ = environment.execute(call['tool'], call['arguments'])
                return {'content': observation}
            return super()._process_output(output, instruction)

        def _execute_code(self, code):
            if 'run_tests' not in environment.tools:
                failure = InvalidOutputError('DyFlow TEST_CODE requested an unavailable tool: run_tests')
                failure.trace = {'code': code, 'calls': calls, 'design_history': workflow.design_history}
                raise failure
            result = environment.observe('run_tests', {'code': code, 'assertions': ''})
            status = 'Error' if 'error' in result or result['status'] != 'passed' else 'Success'
            return status, json.dumps(result, ensure_ascii=False), result

    core['InstructExecutorOperator'] = Operator
    class Workflow(core['WorkflowExecutor']):
        def _design_next_stage(self):
            self.state.original_problem = environment.reset()
            return super()._design_next_stage()

        def _extract_json_from_string(self, text):
            try:
                stage = super()._extract_json_from_string(text)
            except ValueError as error:
                raise InvalidOutputError(f'DyFlow stage design is not valid JSON: {error}') from error
            return validate_output(stage, prompts['stage_schema'], 'stage design')

    workflow = Workflow(environment.reset(), ModelService(role='designer'),
                                        ModelService(role='executor'), save_design_history=True)
    final = workflow.execute()
    if final is None:
        raise TaskLimitError('DyFlow exhausted its declared stage budget without a final answer.')
    answer = parse_output(final, 'final answer')
    try:
        validate_model_arguments(answer, environment.input_schemas['finish'], 'final answer',
                                 {'answer': answer, 'calls': calls, 'design_history': workflow.design_history})
        observation, done = environment.execute('finish', answer)
        if not done:
            failure = InvalidOutputError(observation)
            failure.trace = {'answer': answer, 'calls': calls, 'design_history': workflow.design_history}
            raise failure
    except (ActionError, jsonschema.ValidationError) as error:
        failure = InvalidOutputError(f'DyFlow final answer was rejected: {error}')
        failure.trace = {'answer': answer, 'calls': calls, 'design_history': workflow.design_history}
        raise failure from error
    return {'task_id': task['task_id'], 'answer': environment.answer, 'actions': environment.actions,
            'calls': calls, 'design_history': workflow.design_history, 'state': vars(workflow.state),
            'upstream_commit': settings['upstream_commit'],
            'adaptations': ['shared model and generation policy for every role',
                            'TOOL_CALL operator uses the shared task tools and their original input schemas',
                            'tool and final operators receive immutable task interface metadata',
                            'complete JSON fences are accepted without changing their contents',
                            'TEST_CODE uses the shared read-only sandbox', 'required failures propagate']}
