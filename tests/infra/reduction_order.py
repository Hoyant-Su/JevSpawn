import argparse
from functools import partial
import json
from pathlib import Path

import torch
import torch.distributed as dist

from baselines.common.config import SharedConfig
from baselines.common.parallel_run import initialize_parallel
from jev_spawn.algo.structured import padded
from tests.infra.action_prefix.matched_spawn import requests_for


def compare_reduction(name, records, dtypes, module, inputs, output):
    for dtype in dtypes:
        direct = output.to(dtype).clone()
        permuted = output.flip(0).to(dtype).contiguous()
        dist.all_reduce(direct, group=module._tp_group)
        dist.all_reduce(permuted, group=module._tp_group)
        difference = (direct.float() - permuted.flip(0).float()).abs()
        rounded = (direct.to(output.dtype).float() - permuted.flip(0).to(output.dtype).float()).abs()
        records.append({'module': name, 'shape': list(output.shape), 'dtype': str(dtype),
                        'max_abs_difference': difference.max().item(),
                        'changed_elements': difference.count_nonzero().item(),
                        'rounded_max_abs_difference': rounded.max().item(),
                        'rounded_changed_elements': rounded.count_nonzero().item(),
                        'elements': difference.numel()})


@torch.inference_mode()
def run(settings):
    shared = SharedConfig.load(settings['shared_config'])
    backend, commands, startup = initialize_parallel(shared,
        json.loads(Path(settings['parallel_settings']).read_text()))
    prepared = json.loads(Path(settings['prepared']).read_text())
    workload = prepared['workloads'][settings['workload_index']]
    requests = requests_for(workload, backend, shared, settings)[settings['batch_index']]
    roots = [list(tokens) for tokens in dict.fromkeys(tuple(request.root_tokens) for request in requests)]
    ids, mask = padded(roots, backend.tokenizer.pad_token_id, backend.device, 'left')
    records, handles = [], []
    dtypes = [getattr(torch, name) for name in settings['reduction_dtypes']]
    for name, module in backend.model.named_modules():
        if hasattr(module, '_tp_group'):
            handles.append(module.register_forward_hook(
                partial(compare_reduction, name, records, dtypes), prepend=True))
    backend.model.model(input_ids=ids, attention_mask=mask,
        position_ids=(mask.cumsum(-1) - 1).clamp_min(0), use_cache=True)
    for handle in handles:
        handle.remove()
    output = Path(settings['output'])
    output.mkdir(parents=True, exist_ok=True)
    report = {'settings': settings, 'startup': startup,
        'scope': 'Actual root-prefill projection outputs, reduced twice with original and reversed batch rows. Diagnostic hooks leave original model outputs unchanged; timings are not speed evidence.',
        'task_ids': [request.task_id for request in requests],
        'root_lengths': list(map(len, roots)), 'records': records}
    (output / settings['rank_file'].format(rank=dist.get_rank())).write_text(
        json.dumps(report, indent=2) + '\n')
    print(json.dumps({'rank': dist.get_rank(), 'completed': True,
                      'max_abs_difference': max(row['max_abs_difference'] for row in records)}), flush=True)
    dist.destroy_process_group(commands.control_group)
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    run(json.loads(parser.parse_args().config.read_text()))
