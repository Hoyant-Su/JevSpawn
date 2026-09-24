import ast
import json
from pathlib import Path
import re
from threading import Lock


def validate_selector(text, count):
    match = re.search(r'\{[\s\S]*\}', text)
    if match is None:
        raise ValueError('Ensemble selector produced no JSON object')
    selected = json.loads(match.group())['selected_index']
    if type(selected) is not int or not 1 <= selected <= count:
        raise ValueError('Ensemble selected_index must identify a generated solution')


class SourceInterface(ast.NodeTransformer):
    def visit_ImportFrom(self, node):
        return None if node.level else node

    def visit_Assign(self, node):
        # Validate before upstream's parsing fallback can substitute a solution.
        if any(isinstance(t, ast.Name) and t.id == 'selector_raw' for t in node.targets):
            if isinstance(node.value, ast.Subscript):
                check = ast.parse('validate_selector(selector_raw, len(solutions))').body[0]
                return [node, ast.copy_location(check, node)]
        return self.generic_visit(node)

    def visit_FunctionDef(self, node):
        if node.name == '_execute_code':
            node.body = ast.parse("raise ValueError('TEST_CODE is unavailable in the text MCQ protocol')").body
        return self.generic_visit(node)


def load_core(settings, prompts, service_factory):
    namespace = {'ModelService': service_factory, 'ExecutorLLMClient': object,
                 'validate_selector': validate_selector}
    for name in ['state.py', 'operator.py', 'workflow.py']:
        path = Path(settings['upstream']) / 'dyflow/core' / name
        tree = ast.fix_missing_locations(SourceInterface().visit(ast.parse(path.read_text())))
        exec(compile(tree, str(path), 'exec'), namespace)
    namespace['PROMPT_TEMPLATES'] = prompts['PROMPT_TEMPLATES']
    namespace['DESIGN_STAGE_PROMPT'] = prompts['DESIGN_STAGE_PROMPT']
    original = namespace['InstructExecutorOperator']

    class TextOperator(original):
        def execute(self, state, params):
            instruction = params['instruction_type']
            if instruction not in settings['allowed_operators']:
                raise ValueError('Unsupported text operator: ' + str(instruction))
            if instruction == 'TERMINATE' and not {'final_answer', 'final_answer_key'} & params.keys():
                raise ValueError('TERMINATE requires an actual final answer or state reference')
            return super().execute(state, params)

    namespace['InstructExecutorOperator'] = TextOperator
    return namespace['WorkflowExecutor']


def solve(task, complete, settings, prompts):
    calls = []
    lock = Lock()

    class ModelService:
        def __init__(self, model='gpt-4o-mini', temperature=None, role='summary'):
            self.model = model
            self.temperature = settings['default_temperature'] if temperature is None else temperature
            self.role = role

        def generate(self, prompt, temperature=None, max_tokens=None):
            temperature = self.temperature if temperature is None else temperature
            max_tokens = settings['default_max_tokens'] if max_tokens is None else max_tokens
            record = {'role': self.role, 'upstream_model': self.model, 'prompt': prompt,
                      'temperature': temperature, 'max_tokens': max_tokens}
            with lock:
                record['call_id'] = len(calls)
                calls.append(record)
            response = complete([{'role': 'user', 'content': prompt}], max_tokens, temperature)[0]
            record['response'] = response
            return {'response': response}

    workflow_class = load_core(settings, prompts, ModelService)
    field = task['fields']['q0']
    problem = prompts['problem'].format(
        state=task['state'], question=field['question'],
        options='\n'.join(o['id'] + '. ' + o['description'] for o in field['options']))
    workflow = workflow_class(problem, ModelService(model='gpt-4.1', role='designer'),
                              ModelService(model='phi-4', role='executor'), save_design_history=True)
    final = workflow.execute()
    answer = None
    parse_error = None
    try:
        answer = json.loads(final)['answer']
        if answer not in [o['id'] for o in field['options']]:
            raise ValueError('Final answer is not a listed option ID')
    except (TypeError, ValueError, KeyError) as error:
        answer = None
        parse_error = str(error)
    return {'task_id': task['task_id'], 'answer': answer, 'final_output': final,
            'parse_error': parse_error, 'calls': calls, 'model_calls': len(calls),
            'design_history': workflow.design_history, 'state': vars(workflow.state),
            'upstream_commit': settings['upstream_commit']}
