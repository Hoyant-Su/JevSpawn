import argparse
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace

from transformers import AutoTokenizer

from jev_spawn.schema import CONTROLLER
from methods.evidence_flow.environment import read_jsonl
from methods.evidence_interfaces.inputs import read, write
from methods.source_interfaces.inputs import input_size
from methods.source_interfaces.schema import answer_schema
from methods.source_sets.inputs import fields, final_size, partition
from methods.structured_flow.grammar import SchemaDecoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', type=Path, required=True)
    args = parser.parse_args()
    stage = read(args.stage)
    fixed, prompts = stage['fixed'], read(stage['prompts'])
    tokenizer = AutoTokenizer.from_pretrained(fixed['model_path'])
    backend = SimpleNamespace(tokenizer=tokenizer)
    backend._render = lambda users, system: tokenizer.apply_chat_template(
        [[{'role': 'system', 'content': system}, {'role': 'user', 'content': user}] for user in users],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    CONTROLLER['option_template'] = prompts['option_template']
    configuration = read(Path(fixed['model_path']) / 'config.json')['text_config']
    decoder = SchemaDecoder(tokenizer, [configuration['eos_token_id']], configuration['vocab_size'])
    for contract in [read(stage['plan_schema']), answer_schema(0), answer_schema(fixed['context_tokens'])]:
        decoder.factory(contract, kind='answer')
    rows = []
    tasks = read_jsonl(stage['data']['tasks'])
    previous = read('configs/methods/source_interfaces/stage.json')
    assert stage['data'] == previous['data'] and Path(stage['data']['tasks']).parent.name == 'development'
    for index, task in enumerate(tasks):
        recorded = read(Path(previous['run']) / 'streamed' / 'measured' / f'{index:03d}' / 'outcome.json')['primary']
        assert recorded['task_id'] == task['task_id']
        units = recorded['sources']
        questions = [(value['id'], value['question']) for value in recorded['references']]
        groups = partition(backend, units, questions, fixed, prompts)
        covered = []
        for group in groups:
            candidates = group[0]['options']
            complete = candidates[-1]['source_ids']
            expected = {frozenset(values) for n in range(len(complete) + 1) for values in combinations(complete, n)}
            assert {frozenset(option['source_ids']) for option in candidates} == expected
            assert len(candidates) == len(expected) <= fixed['max_options_per_field']
            assert all(field['options'] == candidates for field in group)
            covered.extend(complete)
        assert covered == [unit['id'] for unit in units]
        lengths = [input_size(backend, group, prompts) for group in groups]
        assert max(lengths) <= fixed['context_tokens']
        refs = [{'id': identity, 'question': question, 'source_ids': covered} for identity, question in questions]
        rows.append({'task_id': task['task_id'], 'source_units': len(units), 'fields': len(questions),
                     'groups': len(groups), 'maximum_input_tokens': max(lengths),
                     'unfiltered_final_tokens': final_size(backend, task['query'], refs,
                                                          {unit['id']: unit for unit in units}, prompts)})
    write(Path(stage['cpu_preflight']), {'tasks': rows, 'grammar_count': 3,
        'scope': 'Real recorded development questions and source units validate complete subset enumeration and context partitioning. No model predictions or quality results are generated or reused as current-run outputs.'})
    print(rows)


if __name__ == '__main__':
    main()
