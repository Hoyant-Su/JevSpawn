import argparse
from copy import deepcopy
from functools import partial
from itertools import count
import json
from pathlib import Path
import time

import jsonschema

from baselines.common.errors import InvalidOutputError
from baselines.common.graph_finite_service import StableGraphFiniteService
from baselines.common.parallel_run import execute_with_runner
from baselines.common.run import read, save
from baselines.common.runtime import InferenceRuntime
from baselines.common.transition_budget import TransitionBudget
from data.task_context import rows
from jev_spawn.infra.configuration import resolve_symbol
from jev_spawn.infra.prompts import load_prompt
from jev_spawn.runtime.branches import Branch, spawn_scored
from jev_spawn.runtime.query_execution import QueryExecution
from jev_spawn.schema.declaration import compile_declaration, declaration_contract
from declaration_history.src.declaration_builder import DeclarationBuilder


def run(specification, output, backend):
    method = read(specification['method'])
    rollout = method['settings']['rollout']
    settings = read(specification['builder'])
    builder_prompts = read(Path(specification['builder']).with_name('prompts.json'))
    inference = read(specification['inference'])
    tasks = []
    for path in specification['task_specifications']:
        source = read(path)
        tasks.extend({**task, 'qualification_source': source} for task in rows(source['tasks']))
    assert len(tasks) == specification['task_count']
    output.mkdir(parents=True, exist_ok=True)
    save(output / 'protocol.json', {'specification': specification, 'builder': settings,
        'builder_prompts': builder_prompts, 'tasks': tasks,
        'shared_config_text': Path(specification['shared_config']).read_text(),
        'scope': 'Typed declaration construction and one scored spawn round with native observations; no task accuracy is evaluated.'})
    runtime = InferenceRuntime(specification['shared_config'],
        partial(StableGraphFiniteService, settings=inference['settings'],
                prompts=load_prompt(inference['prompts'])), backend=backend)
    runtime.service.configure_runtime_contract(method['settings'])
    runtime.service.agent_tokenizers = {task['task_id']: runtime.backend.tokenizer for task in tasks}
    save(output / 'runtime.json', runtime.metadata())
    budget = {'max_new_tokens': runtime.config.generation.max_new_tokens,
              'temperature': runtime.config.generation.temperature}
    contract = declaration_contract(rollout['declaration_schema'], len(runtime.backend.answer_labels))
    contract['properties']['fields']['items']['properties']['values']['minItems'] = specification['minimum_distinct_candidates']
    indexes = {task['task_id']: index for index, task in enumerate(tasks)}

    def execute(task):
        index = indexes[task['task_id']]
        source = task['qualification_source']
        definition = source['environment_execution']
        session = read(source['environment'])
        environment = resolve_symbol(definition['class'])(task, session,
            output / 'tools' / str(index), deadline=partial(runtime.deadlines.remaining, task['task_id']),
            evidence=None, **definition['parameters'])
        transitions = []
        environment = TransitionBudget(environment, runtime.config.runtime.max_turns,
            session['submission_tool'], session['initial_transition_depth'], transitions)
        context = environment.reset()
        tools = environment.display_tool_interface(True)
        answer_schema = environment.display_answer_schema()
        trace = {'task_id': task['task_id'], 'context': context,
                 'public_tools': tools, 'answer_schema': answer_schema}
        started = time.perf_counter()
        phase = 'action_construction'
        try:
            builder = DeclarationBuilder(context, tools, answer_schema,
                service=runtime.service, task_id=task['task_id'], budget=budget,
                settings=settings, prompts=builder_prompts,
                execution=rollout['execution'], trace=trace, feedback={})
            declaration = builder.build_action()
            jsonschema.validate(declaration, contract)
            compiled = compile_declaration(declaration, rollout['execution'])
            trace['compiled_declaration'] = compiled
            roles = [field['question'] for field in declaration['fields']]
            trace['field_counts'] = {'declared': len(roles), 'distinct_roles': len(set(roles)),
                                    'compiled': len(compiled['fields'])}
            trace['action_status'] = 'compiled'
            phase = 'native_spawn'
            execution_context = load_prompt(rollout['prompts'])['execution_context'].format(
                context=context, declaration=json.dumps(compiled, **rollout['execution']['serialization']))
            root = QueryExecution(execution_context, runtime.service, task['task_id'], rollout['execution'])
            identities = (rollout['node_id'].format(index=number)
                          for number in count(rollout['initial_node_index']))
            trace['native_spawn'] = {'parent_computations': root.trace, 'transitions': transitions}
            children = spawn_scored({rollout['root_id']: Branch(root, environment, compiled)},
                rollout['branch_width'], rollout['host_workers'], identities)
            trace['native_spawn']['children'] = {identity: {'trace': child.execution.trace,
                    'done': child.environment.done, 'answer': child.environment.answer}
                    for identity, child in children.items()}
            trace['native_spawn_status'] = 'completed'
            phase = 'answer_construction'
            builder.build_answer()
            jsonschema.validate(declaration, contract)
            trace['compiled_declaration'] = compile_declaration(declaration, rollout['execution'])
            trace['construction_status'] = 'compiled'
        except (AssertionError, KeyError, ValueError, jsonschema.ValidationError, jsonschema.SchemaError) as error:
            trace.update(construction_status='failed', failed_phase=phase,
                         error_type=type(error).__name__, error=str(error))
            raise InvalidOutputError(str(error)) from error
        finally:
            trace['elapsed_seconds'] = time.perf_counter() - started
            save(output / f'construction-{index:05d}.json', trace)
        return {'task_id': task['task_id'], 'answer': None, 'construction_status': 'compiled',
                'declared_fields': len(roles), 'compiled_fields': len(compiled['fields']),
                'generated_tokens': trace['generated_tokens'], 'finite_decisions': trace['finite_decisions']}

    def commit(index, result):
        save(output / f'task-{index:05d}.json', result)
        print(json.dumps(result), flush=True)

    pending = [task for index, task in enumerate(tasks) if not (output / f'task-{index:05d}.json').exists()]
    try:
        results, elapsed = runtime.run(pending, execute,
            lambda index, result: commit(indexes[result['task_id']], result))
        save(output / 'completion.json', {'tasks_this_session': len(results), 'elapsed_seconds': elapsed,
             'results': results, 'task_accuracy_evaluated': False})
    finally:
        runtime.close()
        save(output / 'batches.json', runtime.service.records)
        save(output / 'input_failures.json', runtime.service.input_failures)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--specification', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    specification = read(args.specification)
    execute_with_runner(specification, args.output, read(specification['parallel_settings']), run)


if __name__ == '__main__':
    main()
