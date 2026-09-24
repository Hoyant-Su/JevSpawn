from collections import defaultdict
import json
from pathlib import Path

import jsonschema

from methods.generated_schema.run import measured_call

from project_paths import ROOT
from jev_spawn.infra.prompts import load_prompt


def candidate_space(worker, item):
    source = worker['options']
    if isinstance(source, list):
        return source
    value = item['input']
    for key in source['input_path']:
        value = value[key]
    return value


def compilation_input(job, scope):
    contract = {'instruction': job['instruction'], 'item_interface': job['item_interface'],
                'output_contract': job['output_contract']}
    if scope == 'instance':
        return {'instruction': job['instruction'], 'shared_input': job['shared_input'],
                'item_interface': job['item_interface'], 'item_count': len(job['items']),
                'output_contract': job['output_contract']}
    assert scope == 'contract'
    return {**contract, 'shared_interface': job['shared_interface']}


def compile_programs(backend, jobs, settings):
    directory = ROOT / 'configs/methods/decision_program/schema'
    prompts = load_prompt('configs/methods/decision_program/schema/prompts.json')
    schema = json.loads((directory / 'program.json').read_text())
    contracts = [compilation_input(job, settings['compilation_scope']) for job in jobs]
    serialized = [json.dumps(contract, sort_keys=True, ensure_ascii=False) for contract in contracts]
    unique = list(dict.fromkeys(serialized))
    inputs = [contracts[serialized.index(contract)] for contract in unique]
    binding = [unique.index(contract) for contract in serialized]
    user = [prompts['user'].format(task=json.dumps(task, ensure_ascii=False),
                                   schema=json.dumps(schema, separators=(',', ':')))
            for task in inputs]
    record = measured_call(backend, lambda: backend.generate(
        user, prompts['system'], settings['compiler_tokens']), record_tokens=True)
    programs, failures = {}, []
    for job, index in zip(jobs, binding):
        output = record['result']['texts'][index]
        try:
            program = json.loads(output)
            jsonschema.validate(program, schema)
            reduction = program['reduce']
            assert reduction['operator'] == job['output_contract']['operator'], 'Reducer violates the requested output contract.'
            for item in job['items']:
                candidates = candidate_space(program['worker'], item)
                jsonschema.validate(candidates, schema['$defs']['candidates'])
                ids = [option['id'] for option in candidates]
                assert len(ids) == len(set(ids)), 'Candidate IDs must be unique.'
                if reduction['operator'] == 'rank':
                    assert reduction['score_option'] in ids, 'Ranking score must refer to an available option.'
                else:
                    assert reduction['score_option'] is None, 'Collection does not assign a ranking score.'
        except (ValueError, KeyError, TypeError, jsonschema.ValidationError, AssertionError) as error:
            failures.append({'task_id': job['task_id'], 'error': str(error), 'text': output})
        else:
            programs[job['task_id']] = program
    return {'programs': programs, 'failures': failures, 'generation': record, 'inputs': inputs,
            'root_to_program': dict(zip([job['task_id'] for job in jobs], binding)),
            'compiled_program_count': len(inputs), 'compilation_scope': settings['compilation_scope']}


def instantiate(job, program):
    worker = program['worker']
    return [{'id': item['id'],
             'state': json.dumps({'task': job['shared_input'], 'worker_role': worker['role'],
                                  'item': item['input']}, ensure_ascii=False),
             'question': worker['question'], 'options': candidate_space(worker, item)}
            for item in job['items']]


def compatible_jobs(jobs, programs, batch_size):
    buckets = defaultdict(list)
    for job in jobs:
        program = programs[job['task_id']]
        signature = tuple(len(candidate_space(program['worker'], item)) for item in job['items'])
        buckets[signature].append(job)
    for group in buckets.values():
        for start in range(0, len(group), batch_size):
            yield group[start:start + batch_size]


def reduce_results(program, decisions):
    if program['reduce']['operator'] == 'collect':
        return [{'id': decision['id'], 'choice': decision['choice']} for decision in decisions]
    option = program['reduce']['score_option']
    return [decision['id'] for decision in sorted(
        decisions, key=lambda row: -row['probabilities'][row['option_ids'].index(option)])]
