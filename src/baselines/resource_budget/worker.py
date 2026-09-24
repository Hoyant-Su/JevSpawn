import importlib
import os
import random
import sys
from threading import Lock
import time

import pynvml
import torch

from baselines.formal_choices.run import execute_block, load_protocol, read, save
from baselines.official.model_service import GenerationService
from baselines.resource_budget.policies import AgentPrunePolicy, LatentMASPolicy, rows
from baselines.resource_budget.memory import process_memory
from baselines.single_reasoning.run import execute as execute_single
from jev_spawn.infra.backend import Backend


def allocator_snapshot():
    pynvml.nvmlInit()
    torch.cuda.synchronize()
    physical = process_memory(os.getpid())
    assert physical is not None, 'NVML cannot identify the current model process.'
    result = {**physical, 'allocated_bytes': torch.cuda.memory_allocated(),
              'reserved_bytes': torch.cuda.memory_reserved(),
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'peak_reserved_bytes': torch.cuda.max_memory_reserved()}
    result['physical_minus_reserved_bytes'] = result['physical_bytes'] - result['reserved_bytes']
    pynvml.nvmlShutdown()
    return result


class TrialService(GenerationService):
    def _generate(self, batch):
        try:
            return super()._generate(batch)
        except torch.OutOfMemoryError as error:
            self.report({'event': 'out_of_memory', 'error': str(error)})
            raise


def run(config, output, connection):
    lock = Lock()

    def report(event):
        with lock:
            connection.send(event)

    try:
        execute(config, output, connection, report)
    except torch.OutOfMemoryError as error:
        report({'event': 'out_of_memory', 'error': str(error)})
        raise
    except Exception as error:
        report({'event': 'failed', 'error': type(error).__name__ + ': ' + str(error)})
        raise
    finally:
        connection.close()


def execute(config, output, connection, report):
    study = read(config['study'])
    if config['policy'] in {'formal_choices', 'single_reasoning'}:
        protocol = load_protocol(config['settings'])
    else:
        settings = read(config['settings'])
        tasks, development = rows(settings['tasks']), rows(settings['warmup_tasks'])
        if config['policy'] == 'agentprune':
            source = {row['task_id']: row for row in development}
            warmup = [source[identity] for identity in settings['warmup_task_ids']]
            assert len(warmup) == settings['warmup_task_count']
        else:
            assert config['policy'] == 'latentmas'
            warmup = development
        protocol = {'settings': settings, 'native': read(settings['native_config']),
                    'tasks': tasks, 'warmup': warmup}
    settings, native = protocol['settings'], protocol['native']
    assert native['model_path'] == study['fixed']['model_path']
    assert native['dtype'] == study['fixed']['dtype']
    assert native['batch_size'] == study['fixed']['batch_capacity']
    assert native['max_input_tokens'] == study['fixed']['input_token_limit']
    assert torch.cuda.device_count() == study['fixed']['gpu_count']
    assert torch.cuda.get_device_name(0) == study['fixed']['gpu_model']
    assert len(protocol['tasks']) == settings['task_count']
    assert len({row['task_id'] for row in protocol['tasks']}) == settings['task_count']
    assert not {row['task_id'] for row in protocol['tasks']} & {row['task_id'] for row in protocol['warmup']}
    capacity = int(config['memory_gib'] * 2**30)
    total = torch.cuda.get_device_properties(0).total_memory
    assert 0 < capacity <= total
    torch.cuda.set_per_process_memory_fraction(capacity / total, 0)
    save(output / 'protocol.json', protocol)
    load_started = time.perf_counter()
    backend = Backend(native)
    service = None
    policy = None
    if config['policy'] in {'formal_choices', 'agentprune'}:
        service = TrialService(backend, native['batch_size'], settings['batch_wait_seconds'])
        service.report = report
    if config['policy'] == 'formal_choices':
        sys.path[:0] = settings['python_paths']
        solve = importlib.import_module(settings['adapter_module']).solve
    elif config['policy'] == 'single_reasoning':
        protocol['agent'] = read(settings['agent_config'])
        assert protocol['agent']['batch_size'] == native['batch_size']
        assert protocol['agent']['seed'] == native['seed']
        assert protocol['agent']['max_model_calls_per_task'] == 1
        assert protocol['agent']['max_new_tokens'] == protocol['agent']['max_output_tokens_per_task'] == 2048
    elif config['policy'] == 'agentprune':
        assert settings['batch_size'] == native['batch_size']
        assert settings['max_input_tokens'] == native['max_input_tokens']
        policy = AgentPrunePolicy(backend, settings, service)
    else:
        assert config['policy'] == 'latentmas'
        assert settings['root_batch_size'] == native['batch_size']
        assert settings['context_tokens'] == native['max_input_tokens']
        assert settings['seed'] == native['seed']
        policy = LatentMASPolicy(backend, settings, protocol['tasks'], protocol['warmup'], output)
    save(output / 'effective_protocol.json', protocol)

    def batch(tasks, directory, seed, offset):
        random.seed(seed)
        torch.manual_seed(seed)
        if config['policy'] == 'formal_choices':
            result = execute_block(service, solve, tasks, protocol, directory, seed)
            values = [row['answer'] if row['status'] == 'completed' else None for row in result['results']]
        elif config['policy'] == 'single_reasoning':
            execute_single(backend, tasks, protocol, directory, seed)
            result = read(directory / 'complete.json')
            values = [row['choice'] if row['status'] == 'complete' else None for row in result['results']]
        else:
            values = policy.batch(tasks, directory, seed, offset)
        assert len(values) == len(tasks)
        save(directory / 'memory-snapshot.json', allocator_snapshot())
        return [{'task_id': task['task_id'], 'answer': value,
                 'valid': value in [option['id'] for option in task['fields']['q0']['options']]}
                for task, value in zip(tasks, values)]

    load_seconds = time.perf_counter() - load_started
    warmup_started = time.perf_counter()
    if policy is None:
        batch(protocol['warmup'], output / 'warmup', native['seed'], 0)
    else:
        policy.warmup(protocol['warmup'], output / 'warmup', native['seed'])
    torch.cuda.synchronize()
    report({'event': 'ready', 'loading_seconds': load_seconds,
            'warmup_seconds': time.perf_counter() - warmup_started,
            'allocator_limit_bytes': capacity, 'physical_device_bytes': total,
            'memory_snapshot': allocator_snapshot(),
            'backend': backend.metadata,
            'delivery': 'Atomic original task blocks, including their full execution and persistence.'})
    assert connection.recv() == {'event': 'start'}
    for index, offset in enumerate(range(0, len(protocol['tasks']), native['batch_size'])):
        tasks = protocol['tasks'][offset:offset + native['batch_size']]
        seed = native['seed'] if config['policy'] == 'latentmas' else native['seed'] + index + 1
        report({'event': 'delivery', 'results': batch(tasks, output / f'block-{index:04d}',
                                                      seed, offset)})
    if service is not None:
        service.close()
    report({'event': 'completed'})
