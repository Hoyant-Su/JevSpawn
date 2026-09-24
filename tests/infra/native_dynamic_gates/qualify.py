import argparse
import json
from pathlib import Path
import time

import torch
import torch.distributed as dist
import triton

from baselines.common.config import SharedConfig
from baselines.common.parallel_run import initialize_parallel
from candidate import gates_kernel as dynamic_kernel
from jev_spawn.algo.structured import padded
from jev_spawn.infra.cached_suffix import ragged_suffix
from jev_spawn.infra.qwen35 import gates, gdn, ragged_gdn
from jev_spawn.runtime.native_cache_batch import split_native_cache


@torch.inference_mode()
def run(settings):
    shared = SharedConfig.load(settings['shared_config'])
    backend, commands, startup = initialize_parallel(
        shared, json.loads(Path(settings['parallel_settings']).read_text()))
    destination = Path(settings['output'])
    destination.mkdir(parents=True, exist_ok=True)
    reference = gates.gates_kernel
    rows, checks, known = [], [], set()

    def compare(module, a, b):
        policy = module._execution_settings['gates']
        width = a.shape[-1]
        a_rows, b_rows = a.reshape(-1, width), b.reshape(-1, width)
        signature = (a.numel(), width, a_rows.stride(), b_rows.stride())
        outputs = {}
        row = {'elements': a.numel(), 'shape': list(a.shape),
               'a_stride': list(a_rows.stride()), 'b_stride': list(b_rows.stride()),
               'source': current_source}
        for name, kernel in [('reference', reference), ('dynamic', dynamic_kernel)]:
            g = torch.empty(a.shape, dtype=torch.float32, device=a.device)
            beta = torch.empty(b.shape, dtype=b.dtype, device=b.device)
            arguments = (a_rows, b_rows, module._gdn_decay, module.dt_bias, g, beta,
                         width, a.numel(), *a_rows.stride(), *b_rows.stride(),
                         policy['softplus_threshold'], policy['block_size'])
            grid = (triton.cdiv(a.numel(), policy['block_size']),)
            torch.cuda.synchronize(backend.device)
            started = time.perf_counter()
            binary = kernel[grid](*arguments, **policy['launch'])
            torch.cuda.synchronize(backend.device)
            first = time.perf_counter() - started
            if signature not in known:
                samples = []
                for _ in range(settings['repetitions']):
                    started = time.perf_counter()
                    kernel[grid](*arguments, **policy['launch'])
                    torch.cuda.synchronize(backend.device)
                    samples.append(time.perf_counter() - started)
                row[name] = {'first_invocation_seconds': first,
                             'warm_seconds': samples, 'binary_hash': binary.hash}
            outputs[name] = (g, beta)
        equal = all(torch.equal(left, right) for left, right in
                    zip(outputs['reference'], outputs['dynamic'], strict=True))
        checks.append({'source': current_source, 'layer': module.layer_idx,
                       'shape': list(a.shape), 'exact': equal})
        assert equal, 'Dynamic-length gate changed actual model activations.'
        if signature not in known:
            rows.append(row)
            known.add(signature)
        return outputs['reference']

    gdn.gdn_gates = compare
    ragged_gdn.gdn_gates = compare
    for current_source in settings['sources']:
        requests = json.loads(Path(current_source).read_text())['requests']
        roots = [request['root_tokens'] for request in requests]
        tails = [request['input_ids'][len(root):-1]
                 for request, root in zip(requests, roots, strict=True)]
        ids, mask = padded(roots, backend.tokenizer.pad_token_id, backend.device, 'left')
        output = backend.model.model(input_ids=ids, attention_mask=mask,
            position_ids=(mask.cumsum(-1) - 1).clamp_min(0), use_cache=True)
        states = split_native_cache(output.past_key_values, list(map(len, roots)))
        work = {'computed_input_tokens': 0, 'padded_input_tokens': 0}
        ragged_suffix(backend, states, tails, work)
        report = {'settings': settings, 'startup': startup, 'rows': rows, 'checks': checks,
                  'all_exact': all(check['exact'] for check in checks),
                  'distinct_reference_binaries': len({row['reference']['binary_hash'] for row in rows}),
                  'distinct_dynamic_binaries': len({row['dynamic']['binary_hash'] for row in rows}),
                  'scope': 'Actual model gate inputs, identical gate arithmetic. First-invocation time includes current disk-cache/module-load state; this is not an empty-cache compile benchmark or task-level speedup.'}
        (destination / f'rank-{dist.get_rank()}.json').write_text(json.dumps(report, indent=2) + '\n')
    dist.destroy_process_group(commands.control_group)
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.config.read_text()))
