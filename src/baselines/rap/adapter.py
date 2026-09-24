"""Adapt finite mathematics answers to the original RAP numeric search core."""

import ast
import json
from pathlib import Path
import re

import numpy as np


def original_search(upstream):
    root = Path(upstream)
    namespace = {'QueryLM': object}
    for relative in ['rap/mcts.py', 'rap/gsm8k_mcts.py']:
        path = root / relative
        tree = ast.parse(path.read_text())
        tree.body = [node for node in tree.body
                     if not (isinstance(node, ast.ImportFrom) and node.level)]
        exec(compile(tree, str(path), 'exec'), namespace)
    return namespace['reasoning_mcts_search']


def solve(row, complete, settings, prompts):
    calls = []
    service = complete.func.__self__
    task_id = row['task_id']

    class WorldModel:
        def query_LM(self, prompt, eos_token_id, num_return_sequences=1, do_sample=True, temperature=0.8):
            output = complete([{'role': 'system', 'content': prompts['generate_system']},
                               {'role': 'user', 'content': prompt}],
                              settings['max_new_tokens'], temperature if do_sample else 0,
                              n=num_return_sequences, stop='\n')
            calls.append({'operation': 'generation', 'prompt': prompt, 'outputs': output})
            return [prompt + value for value in output]

        def query_next_token(self, inputs):
            messages = [[{'role': 'system', 'content': prompts['reward_system']},
                         {'role': 'user', 'content': text}] for text in inputs]
            values = service.next_token_probabilities(messages, ['Yes', 'No'], task_id=task_id)
            calls.append({'operation': 'usefulness_reward', 'prompts': inputs, 'probabilities': values})
            return np.asarray(values)

    field = row['fields']['q0']
    options = '\n'.join(f"{index}. {option['description']}" for index, option in enumerate(field['options'], 1))
    question = prompts['question'].format(state=row['state'], question=field['question'], options=options)
    question = ' '.join(question.splitlines())
    root = Path(settings['upstream'])
    reasoning_prompts = json.loads((root / settings['reasoning_prompts']).read_text())
    reward_prompts = json.loads((root / settings['reward_prompts']).read_text())
    parameters = {key: settings[key] for key in ['n_sample_subquestion', 'temperature', 'mcts_rollouts',
                  'w_exp', 'n_sample_confidence', 'max_depth', 'r_alpha', 'r1_default',
                  'speedup_confidence_batch_size']}
    trajectories, tree, snapshots = original_search(root)(
        question, reasoning_prompts, reward_prompts, WorldModel(), eos_token_id=None, **parameters)
    matches = re.findall(prompts['answer_pattern'], trajectories[-1])
    selected = int(matches[-1]) if matches else 0
    answer = field['options'][selected - 1]['id'] if 1 <= selected <= len(field['options']) else None
    return {'task_id': task_id, 'answer': answer, 'trajectories': trajectories,
            'tree': tree, 'rollouts': len(snapshots), 'calls': calls}
