import argparse
import json
from pathlib import Path
from types import MethodType

import torch
from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
from transformers import Qwen3_5TextConfig
from transformers.cache_utils import DynamicCache
from transformers.models.qwen3_5 import modeling_qwen3_5

from jev_spawn.infra.qwen35.gdn import gdn_forward
from jev_spawn.infra.qwen35.norm import gated
from jev_spawn.infra.qwen35.projection import input_projections, pack_input_projections
from jev_spawn.infra.qwen35.ragged_gdn import ragged_gdn_forward
from jev_spawn.runtime.ragged_suffix import RaggedSuffix


def cache_from_states(config, layer_index, conv, recurrent, kernel):
    cache = DynamicCache(config=config)
    layer = cache.layers[layer_index]
    layer.update_conv_state(conv.clone(), conv_kernel_size=kernel)
    layer.update_recurrent_state(recurrent.clone())
    return cache


def compare(actual, expected, tolerance):
    difference = (actual.float() - expected.float()).abs()
    return {'shape': list(actual.shape), 'max_absolute_difference': difference.max().item(),
            'mean_absolute_difference': difference.mean().item(),
            'bitwise_equal': torch.equal(actual.view(torch.uint8), expected.view(torch.uint8)),
            'within_tolerance': torch.allclose(actual, expected, **tolerance)}


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    settings = json.loads(args.config.read_text())
    execution = json.loads(Path(settings['execution_settings']).read_text())
    device, dtype = torch.device(settings['device']), getattr(torch, settings['dtype'])
    torch.cuda.set_device(device)
    torch.manual_seed(settings['seed'])
    text = json.loads(Path(settings['model_config']).read_text())['text_config']
    for name in settings['sharded_head_fields']:
        assert text[name] % settings['world_size'] == 0
        text[name] //= settings['world_size']
    config = Qwen3_5TextConfig(**text)
    module = modeling_qwen3_5.Qwen3_5GatedDeltaNet(config, settings['layer_index']).to(device=device, dtype=dtype).eval()
    module.requires_grad_(False)
    for name, parameter in module.named_parameters():
        parameter.normal_(settings['parameter_mean'], settings['parameter_std'])
        if name in settings['constant_parameters']:
            parameter.fill_(settings['constant_parameters'][name])
    module._execution_settings = execution
    module.register_buffer('_gdn_decay', -module.A_log.float().exp(), persistent=False)
    module.norm._norm_launch = execution['norm_launch']
    module.norm.forward = MethodType(gated, module.norm)
    pack_input_projections(module, execution['gdn_projections'])
    module.input_projections = MethodType(input_projections, module)
    modeling_qwen3_5.torch_chunk_gated_delta_rule = chunk_gated_delta_rule
    modeling_qwen3_5.torch_recurrent_gated_delta_rule = fused_recurrent_gated_delta_rule
    weights = {name: parameter.clone() for name, parameter in module.named_parameters()}
    lengths = settings['lengths']
    hidden = [torch.randn(settings['singleton'], length, module.hidden_size, device=device, dtype=dtype)
              * settings['hidden_std'] for length in lengths]
    conv = [torch.randn(settings['singleton'], module.conv_dim, module.conv_kernel_size, device=device, dtype=dtype)
            * settings['conv_state_std'] for _ in lengths]
    recurrent = [torch.randn(settings['singleton'], module.num_v_heads, module.head_k_dim, module.head_v_dim,
                            device=device, dtype=getattr(torch, settings['recurrent_dtype']))
                 * settings['recurrent_state_std'] for _ in lengths]
    snapshots = [tensor.clone() for tensor in [*hidden, *conv, *recurrent]]
    independent = []
    for values, initial_conv, initial_recurrent in zip(hidden, conv, recurrent, strict=True):
        cache = cache_from_states(config, settings['layer_index'], initial_conv, initial_recurrent, module.conv_kernel_size)
        output = gdn_forward(module, values, cache_params=cache, attention_mask=None)
        layer = cache.layers[settings['layer_index']]
        independent.append((output, layer.conv_states[0].clone(), layer.recurrent_states[0].clone()))
    records = []
    for order in settings['orders']:
        descriptor = RaggedSuffix([lengths[index] for index in order], device)
        padded = hidden[order[0]].new_zeros((len(order), descriptor.width, module.hidden_size))
        for row, index in enumerate(order):
            padded[row:row + settings['singleton'], :lengths[index]].copy_(hidden[index])
        mask = torch.arange(descriptor.width, device=device)[None] < descriptor.lengths[:, None]
        cache = cache_from_states(config, settings['layer_index'], torch.cat([conv[index] for index in order]),
                                  torch.cat([recurrent[index] for index in order]), module.conv_kernel_size)
        output = ragged_gdn_forward(module, padded, cache, mask, descriptor)
        layer = cache.layers[settings['layer_index']]
        for row, index in enumerate(order):
            expected_output, expected_conv, expected_recurrent = independent[index]
            selection = slice(row, row + settings['singleton'])
            records.append({'order': order, 'source_row': index, 'length': lengths[index],
                'hidden': compare(output[selection, :lengths[index]].contiguous(), expected_output.contiguous(), settings['tolerances']['hidden']),
                'conv': compare(layer.conv_states[0][selection].contiguous(), expected_conv.contiguous(), settings['tolerances']['conv']),
                'recurrent': compare(layer.recurrent_states[0][selection].contiguous(), expected_recurrent.contiguous(), settings['tolerances']['recurrent'])})
    unchanged = all(torch.equal(original, snapshot) for original, snapshot in zip([*hidden, *conv, *recurrent], snapshots, strict=True))
    weights_unchanged = all(torch.equal(parameter, weights[name]) for name, parameter in module.named_parameters())
    passed = unchanged and weights_unchanged and all(record[kind]['within_tolerance'] for record in records for kind in settings['tolerances'])
    result = {'scope': 'Synthetic GPU operator test with actual Qwen3.8-27B TP4 local dimensions, random configured weights and native GDN equations. No model predictions, full-model parity, TP collectives or throughput claim.',
              'model_config': settings['model_config'], 'world_size_dimensions': settings['world_size'],
              'dimensions': {'hidden': module.hidden_size, 'key_heads': module.num_k_heads, 'value_heads': module.num_v_heads,
                             'key_head_dim': module.head_k_dim, 'value_head_dim': module.head_v_dim, 'conv_kernel': module.conv_kernel_size},
              'gpu': torch.cuda.get_device_name(device), 'tolerances': settings['tolerances'],
              'source_tensors_unchanged': unchanged, 'weights_unchanged': weights_unchanged,
              'passed': passed, 'records': records}
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'passed': passed, 'comparisons': len(records), 'source_tensors_unchanged': unchanged,
                      'weights_unchanged': weights_unchanged}))
    assert passed


if __name__ == '__main__':
    main()
