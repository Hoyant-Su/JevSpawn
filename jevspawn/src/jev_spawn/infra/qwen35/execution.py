from types import MethodType

import torch
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet, Qwen3_5RMSNorm, Qwen3_5RMSNormGated

from jev_spawn.infra.qwen35.attention import Qwen35Attention
from jev_spawn.infra.qwen35.gdn import gdn_forward
from jev_spawn.infra.qwen35.norm import gated, normalized
from jev_spawn.infra.qwen35.projection import input_projections, pack_input_projections


@torch.no_grad()
def install_qwen35_execution(model, settings):
    if model.training:
        raise ValueError('Optimized Qwen execution requires an evaluation-mode model.')
    language = model.get_submodule(settings['language_model_path'])
    model.requires_grad_(False)
    for layer in language.layers:
        if layer.block_type == 'full_attention':
            layer.self_attn = Qwen35Attention.from_native(layer.self_attn)
            module, names = layer.self_attn, settings['attention_projections']
        elif isinstance(layer.linear_attn, Qwen3_5GatedDeltaNet):
            module, names = layer.linear_attn, settings['gdn_projections']
            module._execution_settings = settings
            module.register_buffer('_gdn_decay', -module.A_log.float().exp(), persistent=False)
            module.forward = MethodType(gdn_forward, module)
        else:
            raise ValueError(f'Unsupported Qwen decoder block: {layer.block_type}')
        pack_input_projections(module, names)
        module.input_projections = MethodType(input_projections, module)
    for module in language.modules():
        if isinstance(module, (Qwen3_5RMSNorm, Qwen3_5RMSNormGated)):
            module._norm_launch = settings['norm_launch']
            module.forward = MethodType(gated if isinstance(module, Qwen3_5RMSNormGated) else normalized, module)
    return model
