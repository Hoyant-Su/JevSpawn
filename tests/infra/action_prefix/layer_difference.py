import argparse
from functools import partial
from importlib import import_module
import json
from pathlib import Path
import re

import torch
import torch.distributed as dist

from baselines.common.config import SharedConfig
from baselines.common.parallel_run import initialize_parallel
from jev_spawn.runtime.prefix_cache import PrefixCache
from tests.infra.action_prefix.first_difference import prepare_requests
from tests.infra.action_prefix.matched_spawn import measure, seed_roots
from tests.infra.action_prefix.verify_post_readout import tensors


@torch.inference_mode()
def run(settings):
    shared = SharedConfig.load(settings['shared_config'])
    backend, commands, startup = initialize_parallel(
        shared, json.loads(Path(settings['parallel_settings']).read_text()))
    requests, inference = prepare_requests(settings, shared, backend)
    definition = settings['implementation']
    cls = getattr(import_module(definition['module']), definition['class'])
    tail = cls(backend, shared.runtime, PrefixCache(shared.runtime.root_batch_size),
               inference['state_copy'], inference['graph_shape'])
    roots = PrefixCache(shared.runtime.root_batch_size)
    measure(backend, tail, roots, [requests], settings)
    lengths = torch.tensor([len(r.admitted.tokens)-len(r.root_tokens)-1 for r in requests],
                           device=backend.device)
    valid = torch.arange(lengths.max(), device=backend.device)[None, :] < lengths[:, None]
    if settings['phase'] == 'decode':
        valid = torch.ones((len(requests), 1), device=backend.device, dtype=torch.bool)
    modules = {name: module for name, module in backend.model.model.named_modules()
               if re.fullmatch(settings['module_pattern'], name)}
    assert modules
    saved, comparisons, saved_cache, cache_comparisons, graph_comparisons = {}, [], {}, [], []
    reference, candidate = settings['orders']
    for order in (reference, candidate):
        roots.clear()
        tail.prefix_cache.clear()
        seed_roots(backend, roots, [requests])
        inverse = torch.tensor([order['indices'].index(i) for i in range(len(requests))],
                               device=backend.device)
        ordered = [requests[i] for i in order['indices']]
        if settings['phase'] == 'decode':
            measure(backend, tail, roots, [ordered], settings)
            graph_logits = tail.last_logits.clone()
            states = [tail.prefix_cache.entries[(tuple(r.admitted.tokens[:-1]),)] for r in ordered]
            for identity, state in zip(order['indices'], states, strict=True):
                for key, value in tensors(state).items():
                    cache_key = (identity, *key)
                    if order == reference:
                        saved_cache[cache_key] = value.clone()
                    else:
                        cache_comparisons.append({'row': identity, 'tensor': list(key),
                            'bitwise_equal': torch.equal(saved_cache[cache_key], value),
                            'max_absolute_error': (saved_cache[cache_key]-value).abs().max().item()})
            decoder = tail.graphs[tail.last_layout['key']]
            load_states, load_tokens = tail.graph_inputs(
                states, [r.admitted.tokens[-1] for r in ordered], tail.last_layout['physical_rows'])
            decoder.load(load_states, load_tokens)

        def observe(name, module, args, output):
            aligned = output.index_select(0, inverse)
            assert aligned.shape[:2] == valid.shape
            if order == reference:
                saved[name] = aligned.clone()
            else:
                left, right = saved[name][valid].float(), aligned[valid].float()
                delta = left-right
                comparisons.append({'module': name, 'shape': list(aligned.shape),
                    'max_absolute_error': delta.abs().max().item(),
                    'mean_absolute_error': delta.abs().mean().item(),
                    'relative_norm_error': (delta.norm()/left.norm()).item(),
                    'bitwise_equal': torch.equal(left, right)})

        hooks = [module.register_forward_hook(partial(observe, name))
                 for name, module in modules.items()]
        if settings['phase'] == 'suffix':
            measure(backend, tail, roots, [ordered], settings)
        else:
            decoder.step()
            current = tail.graph_outputs(decoder.logits, len(requests),
                                         tail.graph_ids[:graph_logits.shape[-1]])
            graph_comparisons.append({'order': order['name'],
                'bitwise_equal': torch.equal(current, graph_logits),
                'max_absolute_error': (current-graph_logits).abs().max().item()})
        for hook in hooks:
            hook.remove()
    output = Path(settings['output'])
    output.mkdir(parents=True, exist_ok=True)
    (output/settings['rank_file'].format(rank=dist.get_rank())).write_text(json.dumps({
        'settings': settings, 'startup': startup, 'comparisons': comparisons,
        'cache_comparisons': cache_comparisons, 'graph_comparisons': graph_comparisons,
        'scope': 'Diagnostic only. Identical roots and real requests; compare valid tokens after inverse row permutation. No speed measurements claimed.'
    }, indent=2)+'\n')
    dist.destroy_process_group(commands.control_group)
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    run(json.loads(parser.parse_args().config.read_text()))
