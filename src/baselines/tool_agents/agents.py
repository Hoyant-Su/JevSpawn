from concurrent.futures import ThreadPoolExecutor
import json
import time

from baselines.tool_agents.dag import execute_plan
from baselines.tool_agents.tools import ActionError, execute


INVALID = (ValueError, TypeError, KeyError, IndexError, SyntaxError, ZeroDivisionError, OverflowError)


def context(task, prompts, trace):
    history = []
    for event in trace:
        if event['event'] == 'generation':
            history.append({'response': event['text']})
        elif event['event'] == 'observation':
            history.append({key: event[key] for key in ['tool', 'node', 'arguments', 'observation'] if key in event})
    field, = task['fields'].values()
    return prompts['context'].format(state=task['state'], question=field['question'],
                                     options=json.dumps(field['options']), tools=json.dumps(prompts['tools']),
                                     history=json.dumps(history, ensure_ascii=False))


def fail(result, error):
    result['status'] = 'failed'
    result['error'] = str(error)
    result['trace'].append({'event': 'failure', 'error': str(error)})


def finish(result, choice, task):
    field, = task['fields'].values()
    if choice not in [option['id'] for option in field['options']]:
        raise ActionError('Final choice is not a supplied option ID.')
    result.update(status='complete', choice=choice)


def generate_batch(generate, active, tasks, results, prompts, system, stage, config):
    eligible = []
    for index in active:
        result = results[index]
        if result['model_calls'] >= config['max_model_calls_per_task']:
            fail(result, 'Shared per-task model-call budget exhausted.')
        elif result['output_tokens'] + config['max_new_tokens'] > config['max_output_tokens_per_task']:
            fail(result, 'Remaining per-task output budget cannot cover the declared generation call.')
        else:
            eligible.append(index)
    active = eligible
    if not active:
        return {}
    output = generate([context(tasks[index], prompts, results[index]['trace']) for index in active],
                      system, config['max_new_tokens'])
    assert output['batch_size'] == len(active)
    assert all(len(output[key]) == len(active) for key in ['texts', 'input_tokens', 'output_tokens', 'truncated'])
    parsed = {}
    for position, index in enumerate(active):
        result = results[index]
        result['model_calls'] += 1
        result['input_tokens'] += output['input_tokens'][position]
        assert output['output_tokens'][position] <= config['max_new_tokens']
        result['output_tokens'] += output['output_tokens'][position]
        assert result['output_tokens'] <= config['max_output_tokens_per_task']
        result['trace'].append({'event': 'generation', 'stage': stage, 'text': output['texts'][position],
                                'input_tokens': output['input_tokens'][position],
                                'output_tokens': output['output_tokens'][position],
                                'batch_size': len(active), 'batch_seconds': output['elapsed_seconds'],
                                'peak_cuda_memory_bytes': output['peak_cuda_memory_bytes'],
                                'truncated': output['truncated'][position]})
        if 'decode' in output:
            result['trace'][-1]['decode'] = output['decode'][position]
        if output['truncated'][position]:
            fail(result, 'Generation reached the configured token limit.')
            continue
        try:
            value = json.loads(output['texts'][position])
            if not isinstance(value, dict) or not isinstance(value['thought'], str):
                raise ActionError('Generated response requires a thought string in a JSON object.')
            parsed[index] = value
        except INVALID as error:
            fail(result, error)
    return parsed


def react(generate, tasks, config, prompts, results):
    system = prompts['react'].replace('{rounds}', str(config['react_max_rounds']))

    def call(action, state):
        started = time.perf_counter()
        observation = execute(action['tool'], action['arguments'], state, config['calculator'])
        return {'event': 'observation', 'tool': action['tool'], 'observation': observation,
                'started': started, 'finished': time.perf_counter()}

    with ThreadPoolExecutor(max_workers=config['tool_concurrency']) as pool:
        for turn in range(config['react_max_rounds']):
            active = [index for index, result in enumerate(results) if result['status'] == 'running']
            outputs = generate_batch(generate, active, tasks, results, prompts, system, f'react/{turn}', config)
            pending = {}
            for index, value in outputs.items():
                result = results[index]
                try:
                    if set(value) != {'thought', 'action'} or set(value['action']) != {'tool', 'arguments'}:
                        raise ActionError('ReAct requires thought and one action with tool and arguments.')
                    action = value['action']
                    if action['tool'] == 'finish':
                        if set(action['arguments']) != {'choice'}:
                            raise ActionError('Finish requires one choice.')
                        finish(result, action['arguments']['choice'], tasks[index])
                    else:
                        result['trace'].append({'event': 'tool_call', 'tool': action['tool'], 'arguments': action['arguments']})
                        pending[index] = pool.submit(call, action, tasks[index]['state'])
                except INVALID as error:
                    fail(result, error)
            for index, future in pending.items():
                try:
                    results[index]['trace'].append(future.result())
                except INVALID as error:
                    fail(results[index], error)
    for result in results:
        if result['status'] == 'running':
            fail(result, 'ReAct turn limit reached without a final choice.')


def compiler(generate, tasks, config, prompts, results):
    planner = prompts['planner'].replace('{nodes}', str(config['compiler_max_nodes']))
    for attempt in range(config['compiler_max_plans']):
        active = [index for index, result in enumerate(results) if result['status'] == 'running']
        plans = generate_batch(generate, active, tasks, results, prompts, planner, f'plan/{attempt}', config)
        for index, plan in plans.items():
            try:
                if set(plan) != {'thought', 'nodes'}:
                    raise ActionError('Planner response requires thought and nodes.')
                execute_plan(plan['nodes'], tasks[index]['state'], config, prompts['tools'], results[index]['trace'])
            except INVALID as error:
                fail(results[index], error)
        active = [index for index, result in enumerate(results) if result['status'] == 'running']
        system = prompts['joiner'].replace('{plans}', str(config['compiler_max_plans'])).replace('{current}', str(attempt + 1))
        joins = generate_batch(generate, active, tasks, results, prompts, system, f'join/{attempt}', config)
        for index, value in joins.items():
            try:
                if set(value) == {'thought', 'choice'}:
                    finish(results[index], value['choice'], tasks[index])
                elif set(value) == {'thought', 'replan'} and isinstance(value['replan'], str) and value['replan'].strip():
                    results[index]['trace'].append({'event': 'replan', 'reason': value['replan']})
                else:
                    raise ActionError('Joiner must return either choice or a nonempty replan request.')
            except INVALID as error:
                fail(results[index], error)
    for result in results:
        if result['status'] == 'running':
            fail(result, 'Compiler plan limit reached without a final choice.')


def reasoning(generate, tasks, config, prompts, results):
    formatted = dict(prompts, context=prompts['reasoning_context'])
    outputs = generate_batch(generate, list(range(len(tasks))), tasks, results, formatted,
                             prompts['reasoning'], 'reasoning', config)
    for index, value in outputs.items():
        try:
            if set(value) != {'thought', 'choice'}:
                raise ActionError('Reasoning response requires thought and choice.')
            finish(results[index], value['choice'], tasks[index])
        except INVALID as error:
            fail(results[index], error)


def run_batch(generate, tasks, method, config, prompts):
    assert 0 < len(tasks) <= config['batch_size']
    assert config['temperature'] == 0
    assert config['max_new_tokens'] * config['max_model_calls_per_task'] <= config['max_output_tokens_per_task']
    assert method in {'react', 'compiler', 'reasoning'}
    if method == 'react':
        assert config['react_max_rounds'] <= config['max_model_calls_per_task']
    if method == 'compiler':
        assert 2 * config['compiler_max_plans'] <= config['max_model_calls_per_task']
    if method == 'reasoning':
        assert config['max_model_calls_per_task'] == 1
    assert all(len(task['fields']) == 1 for task in tasks)
    started = time.perf_counter()
    results = [{'task_id': task['task_id'], 'method': method, 'status': 'running', 'choice': None,
                'model_calls': 0, 'input_tokens': 0, 'output_tokens': 0, 'trace': []} for task in tasks]
    {'react': react, 'compiler': compiler, 'reasoning': reasoning}[method](generate, tasks, config, prompts, results)
    elapsed = time.perf_counter() - started
    for result in results:
        result['tool_calls'] = sum(event['event'] == 'tool_call' for event in result['trace'])
        result['batch_elapsed_seconds'] = elapsed
        result['root_batch_size'] = len(tasks)
    return results
