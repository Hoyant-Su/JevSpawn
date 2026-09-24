import argparse
import ast
import importlib.util
import json
from pathlib import Path
import re
from types import SimpleNamespace
from typing import Dict, List, Union

from data.prepare_bfcl import read_rows
from project_paths import ROOT


def load_definitions(path, namespace, names=None):
    tree = ast.parse(path.read_text(), filename=str(path))
    nodes = [node for node in tree.body
             if isinstance(node, (ast.FunctionDef, ast.Assign))
             and (names is None or getattr(node, 'name', None) in names)]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


def serialize_calls(calls):
    json.dumps(calls, allow_nan=False)
    expressions = []
    for call in calls:
        if not isinstance(call, dict) or len(call) != 1:
            raise ValueError('Each call must contain one function name and its argument mapping.')
        name, arguments = next(iter(call.items()))
        if not all(part.isidentifier() for part in name.split('.')):
            raise ValueError('Function names must be dotted Python identifiers.')
        if not isinstance(arguments, dict) or not all(key.isidentifier() for key in arguments):
            raise ValueError('Call arguments must be a mapping of keyword names.')
        expressions.append(name + '(' + ', '.join(key + '=' + repr(value)
                                                  for key, value in arguments.items()) + ')')
    return '[' + ', '.join(expressions) + ']'


class BFCLScorer:
    def __init__(self, settings):
        self.settings = settings
        directory = ROOT / settings['upstream']
        spec = importlib.util.spec_from_file_location('bfcl_original_enums', directory / 'enums.py')
        enums = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(enums)
        self.language, self.return_format = enums.Language.PYTHON, enums.ReturnFormat.PYTHON
        model_name = settings['model_name']
        namespace = {'re': re, 'Language': enums.Language,
                     'Dict': Dict, 'List': List, 'Union': Union,
                     'MODEL_CONFIG_MAPPING': {model_name: SimpleNamespace(
                         underscore_to_dot=settings['underscore_to_dot'])}}
        for filename in ['type_mappings.py', 'java_type_converter.py', 'js_type_converter.py']:
            load_definitions(directory / filename, namespace)
        self.checker = load_definitions(directory / 'ast_checker.py', namespace)['ast_checker']
        parser_namespace = {'ast': ast, 're': re, 'ReturnFormat': enums.ReturnFormat}
        self.parser = load_definitions(directory / 'model_handler_utils.py', parser_namespace,
                                       settings['parser_functions'])['ast_parse']
        self.allowed_nodes = tuple(getattr(ast, name) for name in settings['allowed_ast_nodes'])

    def parse(self, completion):
        tree = ast.parse(completion.strip().strip("'"), mode='eval')
        function_attributes = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                target = node.func
                while isinstance(target, ast.Attribute):
                    function_attributes.add(id(target))
                    target = target.value
        for node in ast.walk(tree):
            if not isinstance(node, self.allowed_nodes):
                raise ValueError('Unsupported prediction expression: ' + type(node).__name__)
            if isinstance(node, ast.Call) and node.args:
                raise ValueError('BFCL predictions must use keyword arguments.')
            if isinstance(node, ast.Attribute) and id(node) not in function_attributes:
                raise ValueError('Attribute values are outside the literal prediction grammar.')
            if isinstance(node, ast.Dict) and None in node.keys:
                raise ValueError('Dictionary unpacking is outside the literal prediction grammar.')
        # The original decoder's eval branches are unreachable in this literal-only grammar.
        return self.parser(completion, self.return_format, has_tool_call_tag=False)

    def score(self, task, completion, gold):
        assert task['source']['id'] == gold['source_id']
        assert task['task_id'] == gold['task_id']
        assert gold['category'] in self.settings['categories']
        try:
            calls = self.parse(completion)
        except (SyntaxError, ValueError, TypeError, AttributeError, KeyError, IndexError, AssertionError) as error:
            return {'valid': False, 'error_type': 'prediction_parse_error',
                    'error': [str(error)], 'parsed_calls': None}
        result = self.checker(task['input']['function'], calls, gold['ground_truth'],
                              self.language, gold['category'], self.settings['model_name'])
        return {**result, 'parsed_calls': len(calls)}


def evaluate(tasks, labels, predictions, scorer):
    targets = {row['task_id']: row for row in labels}
    outputs = {row['task_id']: row for row in predictions}
    assert len(outputs) == len(predictions) == len(tasks) == len(targets)
    assert outputs.keys() == targets.keys() == {task['task_id'] for task in tasks}
    scores = [{'task_id': task['task_id'],
               **scorer.score(task, outputs[task['task_id']]['completion'], targets[task['task_id']])}
              for task in tasks]
    correct = sum(row['valid'] for row in scores)
    return {'metric': 'bfcl_original_parallel_ast_accuracy', 'tasks': len(tasks), 'correct': correct,
            'accuracy': correct / len(tasks), 'scores': scores,
            'upstream_revision': scorer.settings['revision'],
            'prediction_language': scorer.settings['prediction_language'],
            'scope': scorer.settings['scope']}


def main():
    parser = argparse.ArgumentParser()
    for name in ['config', 'tasks', 'labels', 'predictions', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    settings = json.loads(args.config.read_text())
    scorer = BFCLScorer(settings['evaluation'])
    report = evaluate(read_rows(args.tasks), read_rows(args.labels), read_rows(args.predictions), scorer)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=settings['indent']) + '\n')


if __name__ == '__main__':
    main()
