import argparse
from collections import Counter
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace

from transformers import AutoTokenizer

from jev_spawn.schema import CONTROLLER
from methods.evidence_flow.environment import read_jsonl
from methods.evidence_interfaces.inputs import read, write
from methods.request_routing.inputs import partition_frontier
from methods.source_interfaces.inputs import input_size
from methods.source_sets.inputs import final_size


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
    parent = read(stage['dense_stage'])
    assert stage['data'] == parent['data'] and stage['fixed'] == parent['fixed']
    assert prompts == read(parent['prompts'])
    tasks = read_jsonl(stage['data']['tasks'])
    rows = []
    for index, task in enumerate(tasks):
        outcome = read(Path(parent['run']) / 'streamed' / 'measured' / f'{index:03d}' / 'outcome.json')
        primary = outcome['primary']
        assert primary['task_id'] == task['task_id']
        searches = [entry for entry in read(Path(outcome['attempt']) / 'tools.json')['operations']
                    if entry['operation'] == 'search']
        references = [{'id': reference['id'], 'question': reference['question'],
                       'source_ids': [hit['id'] for hit in search['results']]}
                      for reference, search in zip(primary['references'], searches)]
        assert len(references) == len(searches) == len(primary['references'])
        assert all(reference['question'] == search['query'] for reference, search in zip(references, searches))
        indexed = {unit['id']: unit for unit in primary['sources']}
        groups = partition_frontier(backend, indexed, references, fixed, prompts)
        expected = Counter((r['id'], identity) for r in references for identity in r['source_ids'])
        observed = Counter()
        for group in groups:
            for field in group:
                candidates = field['options'][-1]['source_ids']
                subsets = {frozenset(values) for n in range(len(candidates) + 1)
                           for values in combinations(candidates, n)}
                assert {frozenset(option['source_ids']) for option in field['options']} == subsets
                assert len(field['options']) == len(subsets) <= fixed['max_options_per_field']
                observed.update((field['id'], identity) for identity in candidates)
        assert observed == expected
        assert set(identity for _, identity in observed) == set(indexed)
        lengths = [input_size(backend, group, prompts) for group in groups]
        assert max(lengths) <= fixed['context_tokens']
        rows.append({'task_id': task['task_id'], 'sources': len(indexed), 'request_source_pairs': sum(expected.values()),
                     'groups': len(groups), 'leaf_workers': sum(map(len, groups)),
                     'maximum_input_tokens': max(lengths),
                     'unfiltered_final_tokens': final_size(backend, task['query'], references, indexed, prompts)})
    assert len(rows) == stage['data']['task_count']
    write(Path(stage['cpu_preflight']), {'tasks': rows, 'all_request_source_pairs_preserved': True,
          'scope': 'Actual recorded development retrieval and text only. No model prediction is supplied by this check.'})
    print(rows)


if __name__ == '__main__':
    main()
