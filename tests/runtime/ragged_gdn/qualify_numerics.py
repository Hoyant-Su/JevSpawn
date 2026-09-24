import argparse
import json
from pathlib import Path
from types import MethodType

import torch
from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
from transformers import Qwen3_5TextConfig
from transformers.models.qwen3_5 import modeling_qwen3_5

from jev_spawn.infra.qwen35.gdn import gdn_forward
from jev_spawn.infra.qwen35.norm import gated
from jev_spawn.infra.qwen35.projection import input_projections, pack_input_projections
from jev_spawn.infra.qwen35.ragged_gdn import ragged_gdn_forward
from jev_spawn.runtime.ragged_suffix import RaggedSuffix
from qualify import cache_from_states, compare


def actual_projected_values(module, hidden_states):
    assert hidden_states.shape[:-1] == module._operator_projected_values[0].shape[:-1]
    return module._operator_projected_values


def observe_downstream(module, hidden, cache, mask, descriptor, projected):
    """Control the projection boundary using values computed by the real projection."""
    observed = {}
    original_projection = module.input_projections
    original_conv = modeling_qwen3_5.causal_conv1d_fn
    original_update = modeling_qwen3_5.causal_conv1d_update

    def convolution(*args, **kwargs):
        output = original_conv(*args, **kwargs)
        observed['convolution'] = output[:, :, -hidden.shape[1]:].transpose(1, 2).clone()
        return output

    def convolution_update(*args, **kwargs):
        output = original_update(*args, **kwargs)
        observed['convolution'] = output.transpose(1, 2).clone()
        return output

    def before_norm(norm, args):
        observed['before_norm'] = args[0].reshape(*hidden.shape[:-1], -1).clone()

    def before_output(projection, args):
        observed['before_output'] = args[0].clone()

    module._operator_projected_values = projected
    module.input_projections = MethodType(actual_projected_values, module)
    modeling_qwen3_5.causal_conv1d_fn = convolution
    modeling_qwen3_5.causal_conv1d_update = convolution_update
    norm_hook = module.norm.register_forward_pre_hook(before_norm)
    output_hook = module.out_proj.register_forward_pre_hook(before_output)
    try:
        output = (gdn_forward(module, hidden, cache_params=cache, attention_mask=mask) if descriptor is None else
                  ragged_gdn_forward(module, hidden, cache, mask, descriptor))
    finally:
        norm_hook.remove()
        output_hook.remove()
        module.input_projections = original_projection
        modeling_qwen3_5.causal_conv1d_fn = original_conv
        modeling_qwen3_5.causal_conv1d_update = original_update
    layer = cache.layers[module.layer_idx]
    observed.update(hidden=output, conv=layer.conv_states[0].clone(), recurrent=layer.recurrent_states[0].clone())
    return observed


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    diagnostic = json.loads(args.config.read_text())
    settings = json.loads(Path(diagnostic['operator_fixture']).read_text())
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
    individual_projections = [module.input_projections(values) for values in hidden]
    records = []
    for order in settings['orders']:
        descriptor = RaggedSuffix([lengths[index] for index in order], device)
        padded = hidden[order[0]].new_zeros((len(order), descriptor.width, module.hidden_size))
        for row, index in enumerate(order):
            padded[row:row + settings['singleton'], :lengths[index]].copy_(hidden[index])
        mask = torch.arange(descriptor.width, device=device)[None] < descriptor.lengths[:, None]
        batch_projection = module.input_projections(padded)
        references = []
        for row, index in enumerate(order):
            projected = tuple(value[row:row + settings['singleton'], :lengths[index]] for value in batch_projection)
            cache = cache_from_states(config, settings['layer_index'], conv[index], recurrent[index], module.conv_kernel_size)
            references.append(observe_downstream(module, hidden[index], cache, None, None, projected))
        cache = cache_from_states(config, settings['layer_index'], torch.cat([conv[index] for index in order]),
                                  torch.cat([recurrent[index] for index in order]), module.conv_kernel_size)
        actual = observe_downstream(module, padded, cache, mask, descriptor, batch_projection)
        aligned_reference = actual['before_output'].clone()
        for row, (index, reference) in enumerate(zip(order, references, strict=True)):
            aligned_reference[row:row + settings['singleton'], :lengths[index]].copy_(reference['before_output'])
        same_shape_reference = module.out_proj(aligned_reference)
        same_shape_actual = module.out_proj(actual['before_output'])
        for row, (index, reference) in enumerate(zip(order, references, strict=True)):
            selection = slice(row, row + settings['singleton'])
            record = {'order': order, 'source_row': index, 'length': lengths[index], 'projection_differences': {},
                      'controlled_downstream': {}}
            for name, batched, independent in zip(diagnostic['projection_names'], batch_projection, individual_projections[index], strict=True):
                record['projection_differences'][name] = compare(batched[selection, :lengths[index]].contiguous(), independent.contiguous(), settings['tolerances']['hidden'])
            for name in ('convolution', 'before_norm', 'before_output', 'hidden'):
                record['controlled_downstream'][name] = compare(actual[name][selection, :lengths[index]].contiguous(), reference[name].contiguous(), settings['tolerances']['hidden'])
            for name in ('conv', 'recurrent'):
                record['controlled_downstream'][name] = compare(actual[name][selection].contiguous(), reference[name].contiguous(), settings['tolerances'][name])
            record['controlled_downstream']['same_shape_output_projection'] = compare(
                same_shape_actual[selection, :lengths[index]].contiguous(), same_shape_reference[selection, :lengths[index]].contiguous(), settings['tolerances']['hidden'])
            records.append(record)
    unchanged = all(torch.equal(original, snapshot) for original, snapshot in zip([*hidden, *conv, *recurrent], snapshots, strict=True))
    weights_unchanged = all(torch.equal(parameter, weights[name]) for name, parameter in module.named_parameters())
    result = {'scope': 'Numerical operator-boundary diagnostic on configured synthetic tensors. Controlled projections are freshly computed real batched projections, reused only to isolate downstream equations. No model predictions or speed claim.',
              'operator_fixture': diagnostic['operator_fixture'], 'original_failure_preserved': 'results/infra/ragged_gdn_operator_gpu_001.json',
              'tolerances_unchanged': settings['tolerances'], 'source_tensors_unchanged': unchanged,
              'weights_unchanged': weights_unchanged, 'records': records}
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'comparisons': len(records), 'source_tensors_unchanged': unchanged, 'weights_unchanged': weights_unchanged}))
    assert unchanged and weights_unchanged


if __name__ == '__main__':
    main()
