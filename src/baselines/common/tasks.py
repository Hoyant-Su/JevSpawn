import argparse
from copy import deepcopy
import json
from pathlib import Path

import jsonschema

from baselines.common.resources import ANSWER_PROPERTIES
from project_paths import ROOT
from jev_spawn.infra.prompts import load_prompt
from data.task_context import render, rows


def read(path):
    return json.loads(Path(path).read_text())


def object_schema(properties):
    return {'type': 'object', 'properties': properties, 'required': list(properties),
            'additionalProperties': False}


def normalize(row, specification, instructions, cutoff):
    kind = specification['kind']
    if kind == 'fields':
        properties = {key: {'type': 'string', 'enum': [option['id'] for option in field['options']]}
                      for key, field in row['fields'].items()}
        payload = {key: row[key] for key in ['state', 'fields']}
    elif kind == 'code':
        properties = deepcopy(ANSWER_PROPERTIES['code'])
        payload = {key: row[key] for key in ['prompt', 'entry_point']}
    elif kind == 'ranking':
        identities = [item['document_id'] for item in row['candidates']]
        assert len(identities) == len(set(identities))
        properties = {'ranking': {'type': 'array', 'items': {'type': 'string', 'enum': identities},
                                  'minItems': cutoff, 'maxItems': cutoff, 'uniqueItems': True}}
        payload = {'query': row['query'], 'ranking_cutoff': cutoff,
                   'catalog': [{'id': item['document_id'], 'retrieval_rank': item['retrieval_rank']}
                               for item in row['candidates']]}
    elif kind == 'answer':
        properties = deepcopy(ANSWER_PROPERTIES['answer'])
        payload = {'query': row['query']}
    elif kind == 'grid':
        properties = deepcopy(ANSWER_PROPERTIES['grid'])
        payload = {'puzzle': row['puzzle']}
    elif kind == 'spans':
        properties = deepcopy(ANSWER_PROPERTIES['spans'])
        payload = {'reference': row['source_info'], 'response': row['response']}
    else:
        raise ValueError('Unsupported task interface: ' + kind)
    contract = object_schema(properties)
    jsonschema.Draft202012Validator.check_schema(contract)
    return {'task_id': row['task_id'], 'dataset': specification['dataset'], 'kind': kind,
            'instruction': instructions[kind], 'input': payload, 'answer_schema': contract,
            'source': row}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--matrix', type=Path, required=True)
    args = parser.parse_args()
    matrix = read(args.matrix)
    instructions = load_prompt('configs/baselines/common/schema/tasks.json')
    output = Path(matrix['output'])
    output.mkdir(parents=True, exist_ok=True)
    counts = []
    for specification in matrix['datasets']:
        source = rows(specification['tasks'])
        assert len(source) == specification['task_count']
        assert len({row['task_id'] for row in source}) == len(source)
        tasks = [normalize(row, specification, instructions, matrix['ranking_cutoff']) for row in source]
        assert [task['source'] for task in tasks] == source
        path = output / (specification['dataset'] + '.jsonl')
        path.write_text(''.join(json.dumps(task, ensure_ascii=False) + '\n' for task in tasks))
        counts.append({'dataset': specification['dataset'], 'tasks': len(tasks), 'path': str(path),
                       'task_ids': [task['task_id'] for task in tasks],
                       'fields': sum(len(task['source']['fields']) for task in tasks) if specification['kind'] == 'fields' else None})
    (output / 'index.json').write_text(json.dumps({'datasets': counts, 'labels_loaded': False}, indent=2) + '\n')
    print(json.dumps([{key: value for key, value in row.items() if key != 'task_ids'} for row in counts]))


if __name__ == '__main__':
    main()
