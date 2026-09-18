"""Turn one bulk job into independently scoped, batched specialist workers."""

import argparse
import json
import os
import time
from pathlib import Path

import torch

from jev_spawn.backend import Backend
from jev_spawn.queue import QueuedWorkflow
from jev_spawn.schema import DECISIONS, SPECIALISTS, SYSTEM
from jev_spawn.workflow import Workflow, code_from_response


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


class Journal:
    def __init__(self, path, rank):
        self.path, self.rank, self.counter = path, rank, 0

    def event(self, kind, task=None, agent=None, parent=None, role=None, **payload):
        result = {'event_id': f'{self.rank}:{time.time_ns()}:{self.counter}',
                  'timestamp': time.time(), 'type': kind, 'task_id': task,
                  'agent_id': agent, 'parent_id': parent, 'role': role, 'payload': payload}
        self.counter += 1
        with self.path.open('a') as stream:
            stream.write(json.dumps(result, ensure_ascii=False) + '\n')
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--rank', type=int, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    assert config['mode'] == 'bulk'
    assert config['controller_mode'] in {'direct', 'json'}
    assert config['spawn_policy'] in {'adaptive', 'all', 'single'}
    assert config['routing_mode'] in {'model', 'none'}
    assert config['routing_mode'] != 'none' or config['spawn_policy'] == 'single'
    assert config['schedule'] in {'chunk', 'stage'}
    assert 0 <= args.rank < config['world_size']
    assert min(config['batch_size'], config['controller_batch_size']) > 0
    assert os.environ['CUDA_VISIBLE_DEVICES'] == str(args.rank)
    data = Path(config['tasks_path']).read_bytes()
    rows = [json.loads(line) for line in data.splitlines()]
    selected = rows[config['task_offset']:config['task_offset'] + config['task_count']]
    assert len(selected) == config['task_count']
    assert all(set(task) == {'task_id', 'dataset', 'prompt', 'entry_point'} for task in selected)
    tasks = selected[args.rank::config['world_size']]
    directory = Path(config['run_dir'])
    directory.mkdir(parents=True, exist_ok=True)
    done_path = directory / f'done-{args.rank}.json'
    if done_path.exists():
        return
    write_json(directory / f'run-{args.rank}.json',
               {'config': config, 'rank': args.rank, 'task_ids': [task['task_id'] for task in tasks]})
    backend = Backend(config)
    torch.cuda.synchronize(backend.device)
    memory = {'resident_allocated_bytes': torch.cuda.memory_allocated(backend.device),
              'resident_reserved_bytes': torch.cuda.memory_reserved(backend.device)}
    torch.cuda.reset_peak_memory_stats(backend.device)
    journal = Journal(directory / f'live-{args.rank}.jsonl', args.rank)
    journal.event('run_started', run_id=config['run_id'], model='Qwen/Qwen3.5-4B', mode='bulk',
                  job=config['job'], rank=args.rank, world_size=config['world_size'],
                  batch_size=config['batch_size'], controller_mode=config['controller_mode'],
                  spawn_policy=config['spawn_policy'], routing_mode=config['routing_mode'])
    staged = config['schedule'] == 'stage'
    workflow = (QueuedWorkflow(backend, config, journal, args.rank) if staged
                else Workflow(backend, config, journal, args.rank))
    write_json(directory / f'backend-{args.rank}.json', backend.metadata)
    plan_path = directory / f'plan-{args.rank}.json'
    if plan_path.exists():
        plan = json.loads(plan_path.read_text())
    else:
        routed = config['routing_mode'] == 'model'
        states, events = [], [journal.event('routing_started' if routed else 'dispatch_started',
                                           gpu_id=args.rank, tasks=len(tasks))]
        for start in range(0, len(tasks), config['controller_batch_size']):
            batch = tasks[start:start + config['controller_batch_size']]
            if routed:
                observations = [json.dumps({'job': config['job'], 'subtask': task['prompt']}, ensure_ascii=False) for task in batch]
                result = backend.decide(observations, **DECISIONS['route'], mode=config['controller_mode'], max_new_tokens=config['controller_max_new_tokens'])
                events.append(journal.event('batch_completed', role='controller', gpu_id=args.rank,
                              batch_size=len(batch), duration_seconds=result['elapsed_seconds'],
                              input_tokens=sum(result['input_tokens']), output_tokens=sum(result['output_tokens'])))
            for index, task in enumerate(batch):
                role = result['choices'][index] if routed else 'implement'
                agent_id = f"{task['task_id']}:{role}"
                state = dict(task, role=role, agent_id=agent_id, usage=[], decisions=[], worker_prompt=task['prompt'])
                if routed:
                    decision = {'stage': 'route', 'choice': role, 'options': DECISIONS['route']['options'],
                                'probabilities': result['probabilities'][index], 'input_tokens': result['input_tokens'][index],
                                'probability_status': 'Uncalibrated conditional option scores; null for generated JSON.'}
                    state['decisions'].append(decision)
                    state['worker_prompt'] = f"Specialist assignment:\n{SPECIALISTS[role]}\n\nTask:\n{task['prompt']}"
                    events.append(journal.event('decision', task=task['task_id'], **decision))
                events.append(journal.event('agent_spawned', task['task_id'], agent_id, config['job']['id'], role,
                              prompt=state['worker_prompt'], gpu_id=args.rank))
                states.append(state)
        plan = {'states': states, 'events': events}
        write_json(plan_path, plan)
        print(json.dumps({'rank': args.rank, 'phase': 'spawned', 'agents': len(states)}), flush=True)
    for chunk, start in enumerate(range(0, len(plan['states']), config['batch_size'])):
        prefix = 'draft' if staged else 'chunk'
        path = directory / f'{prefix}-{args.rank}-{chunk:04d}.json'
        if path.exists():
            continue
        states = plan['states'][start:start + config['batch_size']]
        events = [journal.event('agent_started', s['task_id'], s['agent_id'], config['job']['id'], s['role'], gpu_id=args.rank) for s in states]
        result = backend.generate([s['worker_prompt'] for s in states], SYSTEM['implement'], config['max_new_tokens']['implement'])
        events.append(journal.event('batch_completed', role='worker', gpu_id=args.rank,
                      batch_size=len(states), duration_seconds=result['elapsed_seconds'],
                      input_tokens=sum(result['input_tokens']), output_tokens=sum(result['output_tokens'])))
        for index, state in enumerate(states):
            state['solution'] = code_from_response(result['texts'][index])
            state['initial_solution'] = state['solution']
            state['usage'] = [{'role': state['role'], 'input_tokens': result['input_tokens'][index],
                               'output_tokens': result['output_tokens'][index], 'truncated': result['truncated'][index]}]
            events.append(journal.event('agent_completed', state['task_id'], state['agent_id'], config['job']['id'], state['role'],
                          output=result['texts'][index], gpu_id=args.rank, input_tokens=result['input_tokens'][index],
                          output_tokens=result['output_tokens'][index], truncated=result['truncated'][index]))
        if not staged:
            workflow.finish(states, events)
            for state in states:
                events.append(journal.event('task_completed', task=state['task_id'], solution=state['solution'],
                              agent_count=len(state['usage']), dataset=state['dataset']))
        write_json(path, {'states': states, 'events': events})
        print(json.dumps({'rank': args.rank, 'phase': 'drafted' if staged else 'completed',
                          'tasks': min(start + len(states), len(tasks)), 'total': len(tasks)}), flush=True)
    if staged:
        path = directory / f'chunk-{args.rank}-0000.json'
        if not path.exists():
            drafts = [json.loads(p.read_text()) for p in sorted(directory.glob(f'draft-{args.rank}-*.json'))]
            states = [state for draft in drafts for state in draft['states']]
            events = [event for draft in drafts for event in draft['events']]
            workflow.finish(states, events)
            for state in states:
                events.append(journal.event('task_completed', task=state['task_id'], solution=state['solution'],
                              agent_count=len(state['usage']), dataset=state['dataset']))
            write_json(path, {'states': states, 'events': events})
    journal.event('run_completed', rank=args.rank)
    torch.cuda.synchronize(backend.device)
    memory.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(backend.device),
                  peak_reserved_bytes=torch.cuda.max_memory_reserved(backend.device))
    write_json(directory / f'done-{args.rank}.json', {
        'tasks': len(tasks), 'timestamp': time.time(), 'cuda_memory': memory,
        'memory_scope': 'Current rank process after model loading and peak reset; includes resident weights and inference. '
                        'Excludes other CUDA processes and earlier processes before a restart.',
    })


if __name__ == '__main__':
    main()
