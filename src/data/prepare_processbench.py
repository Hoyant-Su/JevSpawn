import argparse
import json
from collections import Counter
from pathlib import Path
from jev_spawn.infra.prompts import load_prompt, resolve_prompts


def prepare(rows, output, split, schema):
    summary = {}
    for mode in ['per_step', 'first_error']:
        tasks, labels = [], []
        for row in rows:
            count, first_error = len(row['steps']), row['label']
            assert count + 1 <= schema['max_first_error_options'], row['id']
            assert -1 <= first_error < count, row['id']
            paragraphs = [load_prompt('src/data/prepare_processbench.py#paragraph_template').format(index=index, step=step)
                          for index, step in enumerate(row['steps'])]
            state = schema['state'].format(problem=row['problem'], solution='\n\n'.join(paragraphs))
            fields, answers, question_ids = {}, {}, {}
            if mode == 'per_step':
                for index in range(count):
                    name = f'q{index}'
                    fields[name] = {'id': name, 'question': schema['step']['question'].format(index=index),
                                    'options': schema['step']['options']}
                    answers[name] = None
                    if first_error == -1 or index < first_error:
                        answers[name] = 'correct'
                    elif index == first_error:
                        answers[name] = 'incorrect'
                    question_ids[name] = f"{row['id']}/step/{index}"
            else:
                options = [schema['first_error']['correct_option']] + [
                    {'id': str(index), 'description': schema['first_error']['error_description'].format(index=index)}
                    for index in range(count)]
                fields['q0'] = {'id': 'q0', 'question': schema['first_error']['question'], 'options': options}
                answers['q0'], question_ids['q0'] = str(first_error), row['id']
            task_id = f"processbench_gsm8k/{row['id']}"
            tasks.append({'task_id': task_id, 'dataset': 'processbench_gsm8k', 'state': state, 'fields': fields})
            labels.append({'task_id': task_id, 'labels': answers, 'question_ids': question_ids,
                           'first_error': first_error, 'step_count': count})
        directory = output / split / mode
        directory.mkdir(parents=True, exist_ok=True)
        for name, values in [('tasks', tasks), ('labels', labels)]:
            (directory / f'{name}.jsonl').write_text(''.join(json.dumps(value, ensure_ascii=False) + '\n' for value in values))
        summary[mode] = {'tasks': len(tasks), 'decisions': sum(len(task['fields']) for task in tasks),
                         'labelled_fields': sum(value is not None for row in labels for value in row['labels'].values()),
                         'tasks_file': str(directory / 'tasks.jsonl'), 'labels_file': str(directory / 'labels.jsonl')}
    summary.update(error_solutions=sum(row['label'] != -1 for row in rows),
                   correct_solutions=sum(row['label'] == -1 for row in rows),
                   steps_per_solution=dict(sorted(Counter(len(row['steps']) for row in rows).items())))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--schema', type=Path, required=True)
    args = parser.parse_args()
    schema = resolve_prompts(json.loads(args.schema.read_text()))
    evaluation = json.loads((args.source / 'processbench_gsm8k.json').read_text())
    feasibility = json.loads((args.source / 'processbench_math.json').read_text())[:schema['feasibility_count']]
    summary = {split: prepare(rows, args.output, split, schema)
               for split, rows in [('evaluation', evaluation), ('feasibility', feasibility)]}
    summary.update({
        'name': 'ProcessBench GSM8K first-error identification',
        'sources': ['https://github.com/QwenLM/ProcessBench',
                    'https://huggingface.co/datasets/Qwen/ProcessBench',
                    'https://huggingface.co/datasets/Qwen/ProcessBench/resolve/main/gsm8k.json',
                    'https://huggingface.co/datasets/Qwen/ProcessBench/resolve/main/math.json',
                    'https://github.com/QwenLM/ProcessBench/blob/main/code/run_eval.py'],
        'scope': 'All 400 GSM8K solutions for evaluation; fixed first 16 solutions of the separate MATH subset for feasibility. Every original step is preserved.',
        'formulations': {
            'per_step': 'Independent correct/incorrect decisions for every step, each seeing the same complete original solution. Return the smallest predicted incorrect index, or -1 if all steps are predicted correct.',
            'first_error': 'One categorical selection among -1 and every original zero-based step index, with the same complete original solution as context.'},
        'metrics': {'first_error_accuracy': 'Exact predicted index match across all solutions.',
                    'error_accuracy': 'Exact predicted index match among solutions with at least one error.',
                    'correct_accuracy': 'Fraction of correct solutions predicted -1.',
                    'processbench_f1': 'Harmonic mean of error_accuracy and correct_accuracy, matching the official evaluation formula.',
                    'local_step_accuracy': 'Accuracy on steps preceding the first error and on the first error itself; all steps labelled correct when the solution label is -1. Later steps have null labels and are excluded.'},
        'limits': ['Per-step parallel verification and categorical direct selection are adaptations, not the official generated-critique baseline.',
                   'All fields see the complete solution; later text may influence an earlier-step judgment.',
                   'Only the first error is annotated. No correctness label is inferred for subsequent steps.',
                   'final_answer_correct and generator are not supplied to the model.',
                   'ProcessBench F1 is a harmonic mean of two subgroup accuracies, not ordinary classification F1.']})
    (args.output / 'dataset_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
