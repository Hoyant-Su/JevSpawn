from collections import Counter
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def main():
    directory = Path('methods/posterior_messages')
    stage = read(directory / 'stage.json')
    collections = [json.loads(line) for line in Path(stage['data']).read_text().splitlines()]
    prepared, diagnostics = [], []
    for index, collection in enumerate(collections):
        outcome = read(Path(stage['source']) / f'{index:03d}' / 'outcome.json')
        primary = outcome['primary']
        assert primary['task_id'] == collection['task_id']
        definitions = primary['rounds'][0]['questions']
        identifiers = [field['id'] for field in definitions]
        values = [{} for _ in collection['candidates']]
        calls = [json.loads(line) for line in (Path(outcome['attempt']) / 'calls.jsonl').read_text().splitlines()]
        for call in calls:
            if call['kind'] != 'worker':
                continue
            for identity, group in zip(call['context']['documents'], call['result']['groups']):
                for result in group:
                    if result['id'] in identifiers:
                        assert result['id'] not in values[identity]
                        values[identity][result['id']] = result
        documents = []
        for ordinal, (document, value) in enumerate(zip(collection['candidates'], values)):
            assert set(value) == set(identifiers)
            results = []
            for field in definitions:
                result = value[field['id']]
                assert result['option_ids'] == [option['id'] for option in field['options']]
                probabilities = result['probabilities']
                assert len(probabilities) == len(result['option_ids'])
                assert abs(sum(probabilities) - 1) <= len(probabilities) * 2**-23
                assert result['choice'] == result['option_ids'][max(range(len(probabilities)), key=probabilities.__getitem__)]
                results.append({'field_id': field['id'], 'choice': result['choice'],
                                'option_ids': result['option_ids'], 'probabilities': probabilities})
            documents.append({'document_id': document['document_id'], 'ordinal': ordinal, 'results': results})
        counts = Counter(tuple(value[field]['choice'] for field in identifiers) for value in values)
        diagnostics.append({'task_id': collection['task_id'], 'documents': len(documents),
                            'fields': len(definitions), 'unique_winner_messages': len(counts),
                            'largest_identical_message_group': max(counts.values())})
        prepared.append({'task_id': collection['task_id'], 'query': collection['query'],
                         'fields': definitions, 'documents': documents})
    assert len(prepared) == stage['fixed']['queries']
    assert all(len(row['documents']) == stage['fixed']['documents_per_query'] for row in prepared)
    (directory / 'inputs.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in prepared))
    summary = {'queries': diagnostics, 'source': stage['source'], 'labels_read': False,
               'scope': 'Exact extraction of real first-round worker outputs. No model predictions are synthesized.'}
    (directory / 'input_diagnostics.json').write_text(json.dumps(summary, indent=2) + '\n')
    stage.update(status='inputs_prepared_not_launched', inputs='data/methods/posterior_messages/inputs.jsonl',
                 input_diagnostics='results/methods/posterior_messages/input_diagnostics.json')
    (directory / 'stage.json').write_text(json.dumps(stage, indent=2) + '\n')
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
