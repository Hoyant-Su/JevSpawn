"""Adapt finite-answer tasks to the original Tree of Thoughts search."""

import ast
from pathlib import Path
import re
from types import SimpleNamespace


class ChoiceTask:
    def __init__(self, row, settings, prompts, vote_parser):
        self.row, self.prompts = row, prompts
        self.steps = settings['steps']
        self.stops = [None] * self.steps
        self.vote_outputs_unwrap = vote_parser

    def get_input(self, index):
        field = self.row['fields']['q0']
        options = '\n'.join(option['id'] + '. ' + option['description'] for option in field['options'])
        return self.prompts['question'].format(state=self.row['state'],
                                               question=field['question'], options=options)

    def cot_prompt_wrap(self, x, y=''):
        return self.prompts['generate'].format(question=x, previous=y)

    def vote_prompt_wrap(self, x, ys):
        candidates = '\n\n'.join(f'Choice {index}:\n{y}' for index, y in enumerate(ys, 1))
        return self.prompts['vote'].format(question=x, candidates=candidates)


def original_search(upstream, gpt):
    path = Path(upstream) / 'src/tot/methods/bfs.py'
    tree = ast.parse(path.read_text())
    tree.body = [node for node in tree.body
                 if not (isinstance(node, ast.ImportFrom) and node.module == 'tot.models')]
    namespace = {'gpt': gpt}
    exec(compile(tree, str(path), 'exec'), namespace)
    return namespace['solve']


def original_vote_parser(upstream):
    path = Path(upstream) / 'src/tot/tasks/text.py'
    tree = ast.parse(path.read_text())
    task = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'TextTask')
    task.bases = []
    task.body = [node for node in task.body
                 if isinstance(node, ast.FunctionDef) and node.name == 'vote_outputs_unwrap']
    namespace = {'re': re}
    exec(compile(ast.Module(body=[task], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['TextTask'].vote_outputs_unwrap


def solve(row, complete, settings, prompts):
    calls = []

    def gpt(prompt, model=None, temperature=0, max_tokens=None, n=1, stop=None):
        output = complete([{'role': 'user', 'content': prompt}], settings['max_new_tokens'],
                          temperature, n=n, stop=stop)
        calls.append({'prompt': prompt, 'temperature': temperature, 'n': n,
                      'stop': stop, 'outputs': output})
        return output

    task = ChoiceTask(row, settings, prompts, original_vote_parser(settings['upstream']))
    args = SimpleNamespace(backend=settings['model_name'], temperature=settings['temperature'],
                           method_generate='sample', method_evaluate='vote', method_select='greedy',
                           prompt_sample='cot', n_generate_sample=settings['n_generate_sample'],
                           n_evaluate_sample=settings['n_evaluate_sample'],
                           n_select_sample=settings['n_select_sample'])
    candidates, info = original_search(settings['upstream'], gpt)(args, task, 0, to_print=False)
    matches = re.findall(prompts['answer_pattern'], candidates[0])
    return {'task_id': row['task_id'], 'answer': matches[-1] if matches else None,
            'candidates': candidates, 'search': info, 'calls': calls}
