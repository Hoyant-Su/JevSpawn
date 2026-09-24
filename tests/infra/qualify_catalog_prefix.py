import argparse
import json
from pathlib import Path
import time

import torch
import torch.distributed as dist

from baselines.common.config import SharedConfig
from baselines.common.parallel_run import initialize_parallel
from baselines.common.parallel_service import ParallelPrefixCache
from catalog_frontier import admission, save
from jev_spawn.algo.structured import common_prefix
from jev_spawn.infra.readout_labels import AdmittedPrompt
from jev_spawn.runtime.prefix_cache import PrefixCache
from jev_spawn.schema import CONTROLLER, controller_prompts
from methods.program_execution.grouped import score_grouped




def run(settings):
    shared = SharedConfig.load(settings['shared_config'])
    parallel = json.loads(Path(settings['parallel_settings']).read_text())
    backend, commands, startup = initialize_parallel(shared, parallel)
    feasibility = json.loads(Path(settings['feasibility_output']).read_text())
    prefix_plan = json.loads(Path(settings['prefix_plan_output']).read_text())
    assert feasibility['within_context'] and feasibility['all_fields_preserved']
    assert list(backend.answer_labels)==feasibility['answer_labels']
    assert list(backend.answer_label_ids)==feasibility['answer_label_ids']
    output = Path(settings['output'])
    templates = json.loads(Path(settings['templates']).read_text())
    variants = list(dict.fromkeys(phase['variant']for phase in settings['phases']))
    caches = {variant:{name:ParallelPrefixCache(PrefixCache(settings['cache_capacity']),commands)
                     for name in ['state','base']}for variant in variants}
    if commands.is_leader:
        save(output/'execution.json',{'settings':settings,'startup':startup})
    records=[]

    @torch.inference_mode()
    def score(payload):
        variant, group = payload['variant'], payload['group']
        CONTROLLER.clear()
        CONTROLLER.update(group['controller'])
        if payload['clear']:
            for cache in caches[variant].values():
                cache.clear()
        torch.cuda.synchronize(backend.device)
        started=time.perf_counter()
        result=score_grouped(backend,[group['fields']],settings['mode'],
            prefix_cache=caches[variant]['state'],base_prefix_cache=caches[variant]['base'],
            base_prefix_length=payload['base_length'],admitted_prompts=payload['admitted'])
        torch.cuda.synchronize(backend.device)
        local=time.perf_counter()-started
        metrics=torch.tensor([local,*result['timings'].values()],device=backend.device,dtype=torch.float64)
        dist.all_reduce(metrics,op=dist.ReduceOp.MAX)
        values=metrics.tolist()
        record={'phase':payload['phase'],'variant':variant,'frontier_id':group['frontier_id'],
            'group_start':group['start'],'rank':dist.get_rank(),'local_scoring_seconds':local,
            'rankmax_scoring_seconds':values[0],'rankmax_phase_seconds':dict(zip(result['timings'],values[1:])),
            'allocated_bytes':torch.cuda.memory_allocated(backend.device),
            'reserved_bytes':torch.cuda.memory_reserved(backend.device),'result':result}
        save(output/f"rank{dist.get_rank()}-{payload['phase']}-{group['frontier_id']}-{group['start']}.json",record)
        records.append(record)
        return record

    commands.register(settings['command'],score)
    if commands.is_leader:
        try:
            for phase in settings['phases']:
                clear=phase['clear']
                for group in feasibility['groups']:
                    started=time.perf_counter()
                    payload=admission(backend,group,phase['representation'],templates,settings)
                    if phase['prefix_mode'] == 'deepest_common':
                        payload['base_length'] = prefix_plan['frontiers'][group['frontier_id']]['corrected_base_length']
                    payload['variant'] = phase['variant']
                    admitted=time.perf_counter()-started
                    payload.update(phase=phase['name'],clear=clear)
                    record=commands.call(settings['command'],payload)
                    clear=False
                    record.update(admission_seconds=admitted,inclusive_seconds=time.perf_counter()-started)
                    save(output/f"{phase['name']}-{group['frontier_id']}-{group['start']}.json",record)
            save(output/'completion.json',{'settings':settings,'records':records,'prefix_plan':prefix_plan})
            print(json.dumps({'completed_cohorts':len(records)}),flush=True)
        finally:
            commands.finish()
    else:
        commands.serve()
    for bank in caches.values():
        for cache in bank.values():cache.clear()
    dist.destroy_process_group(commands.control_group)
    dist.destroy_process_group()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args()
    run(json.loads(args.config.read_text()))
