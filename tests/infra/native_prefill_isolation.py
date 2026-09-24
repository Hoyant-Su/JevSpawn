import argparse
import inspect
import json
from pathlib import Path
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache
from transformers.models.qwen3_5 import modeling_qwen3_5

from baselines.common.config import SharedConfig
from baselines.common.parallel_run import initialize_parallel
from jev_spawn.infra.backend import KERNELS, STOCK_KERNELS, set_kernel


def callable_identity(function):
    chain = []
    current = function
    while True:
        closure = inspect.getclosurevars(current).nonlocals
        chain.append({'module': current.__module__, 'name': current.__qualname__,
            'code_file': current.__code__.co_filename, 'code_line': current.__code__.co_firstlineno,
            'callable_closure': {key: {'module': value.__module__, 'name': value.__qualname__,
                'code_file': value.__code__.co_filename, 'code_line': value.__code__.co_firstlineno}
                for key, value in closure.items() if inspect.isfunction(value)}})
        if not hasattr(current, '__wrapped__'):
            return chain
        current = current.__wrapped__


@torch.inference_mode()
def main(config):
    source = json.loads(Path(config['source_result']).read_text())
    recorded, = [row for row in source['records'] if row['turn'] == config['diagnostic_turn']]
    diagnostic = recorded['diagnostic']
    shared = SharedConfig.load(config['shared_config'])
    backend, commands, startup = initialize_parallel(shared,
        json.loads(Path(config['parallel_settings']).read_text()))
    native_torch = tuple(inspect.unwrap(function) for function in STOCK_KERNELS)
    assert all(function.__code__.co_filename == modeling_qwen3_5.__file__ for function in native_torch)
    implementations = {**KERNELS, 'torch_native': native_torch}
    ids = torch.tensor(diagnostic['input_token_ids'], device=backend.device, dtype=torch.long)
    mask = torch.tensor(diagnostic['attention_mask'], device=backend.device, dtype=torch.long)
    assert mask.sum(-1).tolist() == diagnostic['input_token_counts']
    native_ids = recorded['rows'][0]['branch_token_ids']
    assert all(row['branch_token_ids'] == native_ids for row in recorded['rows'])
    weight = backend.selected_output_weights(tuple(native_ids)).float()
    cached = torch.tensor([row['cached_logits'] for row in recorded['rows']], device=backend.device)
    original = torch.tensor([row['fresh_logits'] for row in recorded['rows']], device=backend.device)
    layer_outputs, reference_layers, reports = [], None, []

    def observe(module, arguments, output):
        layer_outputs.append(output[:, -1].float().clone())

    layers = backend.model.model.language_model.layers
    hooks = [layer.register_forward_hook(observe) for layer in layers]
    output = Path(config['output'])
    output.mkdir(parents=True, exist_ok=True)
    for mode in config['modes']:
        selected = implementations[mode['kernel']]
        modeling_qwen3_5.torch_chunk_gated_delta_rule, modeling_qwen3_5.torch_recurrent_gated_delta_rule = selected
        identities = [callable_identity(function) for function in selected]
        cache = DynamicCache(config=backend.cache_config, offloading=False) if mode['use_cache'] else None
        layer_outputs.clear()
        torch.cuda.synchronize(backend.device)
        started = time.perf_counter()
        result = backend.model.model(input_ids=ids, attention_mask=mask,
            position_ids=(mask.cumsum(-1) - 1).clamp_min(0),
            past_key_values=cache, use_cache=mode['use_cache'])
        logits = F.linear(result.last_hidden_state[:, -1].float(), weight)
        torch.cuda.synchronize(backend.device)
        elapsed = time.perf_counter() - started
        assert len(layer_outputs) == len(layers)
        if reference_layers is None:
            reference_layers = list(layer_outputs)
        differences = [{'layer': index, 'type': layer.block_type,
            'max_absolute_difference': (current-reference).abs().amax(-1).tolist(),
            'mean_absolute_difference': (current-reference).abs().mean(-1).tolist()}
            for index, (layer, current, reference) in enumerate(zip(layers, layer_outputs, reference_layers, strict=True))]
        reports.append({'mode': mode, 'actual_callable_identities': identities,
            'diagnostic_seconds': elapsed, 'branch_logits': logits.tolist(),
            'chosen_native_tokens': [native_ids[index] for index in logits.argmax(-1).tolist()],
            'maxdiff_recorded_cached': (logits-cached).abs().amax(-1).tolist(),
            'maxdiff_recorded_fresh': (logits-original).abs().amax(-1).tolist(),
            'layer_differences_from_first_mode': differences})
        (output / f'rank-{dist.get_rank()}.json').write_text(json.dumps({
            'scope': config['scope'], 'startup': startup, 'source_result': config['source_result'],
            'turn': config['diagnostic_turn'], 'input_token_ids': diagnostic['input_token_ids'],
            'attention_mask': diagnostic['attention_mask'], 'branch_token_ids': native_ids,
            'records': reports}, **config['serialization']) + '\n')
        if commands.is_leader:
            print(json.dumps({key: value for key, value in reports[-1].items()
                              if key != 'layer_differences_from_first_mode'}), flush=True)
        del result, logits, cache
    for hook in hooks:
        hook.remove()
    set_kernel(shared.model.kernel)
    dist.destroy_process_group(commands.control_group)
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    main(json.loads(parser.parse_args().config.read_text()))
