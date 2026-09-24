import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from causal_conv1d import causal_conv1d_update
from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule
from torch.nn import functional as F
from transformers import StaticCache

from baselines.common.config import SharedConfig
from baselines.common.environment import TaskEnvironment
from baselines.common.tasks import read, rows
from baselines.latentmas.adapter import ModelAdapter
from baselines.latentmas.common_service import LatentTransport, role_messages
from baselines.latentmas.native_hybrid_transport import padding_segments
from baselines.latentmas.numerics import difference
from jev_spawn.infra.backend import Backend
from jev_spawn.infra.prompts import load_prompt
from methods.evidence_flow.environment import EvidenceEnvironment


def compare(left, right, mask):
    return difference(left[mask.bool()], right[mask.bool()])


def project(module, hidden, segments):
    return {name: torch.cat([projection(hidden[:, start:end]) for start, end in segments], dim=1)
            for name, projection in {'qkv': module.in_proj_qkv, 'a': module.in_proj_a,
                                     'b': module.in_proj_b, 'z': module.in_proj_z}.items()}


def convolve(module, projected, initial, mask, segments):
    state, outputs = initial.clone(), []
    for start, end in segments:
        inactive = ~mask[:, start].bool()
        saved = state[inactive].clone()
        output = causal_conv1d_update(projected[:, start:end].transpose(1, 2), state,
            module.conv1d.weight.squeeze(1), module.conv1d.bias, module.activation)
        state[inactive] = saved
        outputs.append(output.transpose(1, 2))
    return torch.cat(outputs, dim=1), state


def recurrent(module, projected, convolved, initial, mask, segments):
    batch, length, _ = convolved.shape
    query, key, value = torch.split(convolved, [module.key_dim, module.key_dim, module.value_dim], dim=-1)
    query = query.reshape(batch, length, module.num_k_heads, module.head_k_dim)
    key = key.reshape(batch, length, module.num_k_heads, module.head_k_dim)
    value = value.reshape(batch, length, module.num_v_heads, module.head_v_dim)
    query = query.repeat_interleave(module.num_v_heads // module.num_k_heads, dim=2)
    key = key.repeat_interleave(module.num_v_heads // module.num_k_heads, dim=2)
    beta = projected['b'].sigmoid()
    decay = -module.A_log.float().exp() * F.softplus(projected['a'].float() + module.dt_bias)
    state, outputs = initial.clone(), []
    for start, end in segments:
        inactive = ~mask[:, start].bool()
        saved = state[inactive].clone()
        output, state = fused_recurrent_gated_delta_rule(query[:, start:end], key[:, start:end], value[:, start:end],
            g=decay[:, start:end], beta=beta[:, start:end], initial_state=state,
            output_final_state=True, use_qk_l2norm_in_kernel=True)
        state[inactive] = saved
        outputs.append(output)
    return torch.cat(outputs, dim=1), state


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config = read(args.config)
    shared = SharedConfig.load(config['shared_config'])
    method, tools = read(config['method']), read(config['environment'])
    tasks = rows(config['tasks'])[config['task_start']:config['task_start'] + config['task_count']]
    assert len(tasks) == shared.runtime.batch_size == config['task_count']
    args.output.mkdir(parents=True, exist_ok=False)
    backend = Backend(shared.backend())
    prompts = load_prompt(method['prompts'])
    evidence = EvidenceEnvironment(**tools['evidence'])
    roles = []
    for task in tasks:
        environment = TaskEnvironment(task, tools, args.output / 'tools' / task['task_id'],
                                      deadline=lambda: None, evidence=evidence)
        messages = [{'role': 'system', 'content': prompts['system']},
                    {'role': 'user', 'content': environment.reset()}]
        roles.append(role_messages(messages, prompts, shared.model.path))
    batches = []
    for role in range(2):
        rendered = [backend.tokenizer.apply_chat_template(messages[role], tokenize=False,
                    add_generation_prompt=True, enable_thinking=False) for messages in roles]
        batches.append(backend.tokenizer(rendered, padding=True, add_special_tokens=False,
                                        return_tensors='pt', truncation=False).to(backend.device))
    planner, critic = batches
    prefix = config['critic_prefix_tokens']
    required = planner['input_ids'].shape[1] + method['settings']['latent_steps'] + prefix
    block = shared.runtime.graph_cache_block_tokens
    cache = StaticCache(config=backend.model.config, max_cache_len=(required + block - 1) // block * block)
    transport = LatentTransport(backend, cache, lambda: None)
    wrapper = ModelAdapter(backend, SimpleNamespace(latent_space_realign=True))
    wrapper.model = transport
    wrapper.generate_latent_batch(**planner, latent_steps=method['settings']['latent_steps'], past_key_values=None)
    decoder_layer = transport.trunk.layers[0]
    module = decoder_layer.linear_attn
    mask = critic['attention_mask'][:, :prefix]
    hidden = decoder_layer.input_layernorm(backend.model.get_input_embeddings()(critic['input_ids'][:, :prefix]))
    hidden = hidden * mask[:, :, None]
    token_segments = [(index, index + 1) for index in range(prefix)]
    chunk_segments = [(start, min(end, prefix)) for start, end in padding_segments(critic['attention_mask']) if start < prefix]
    token = project(module, hidden, token_segments)
    chunk = project(module, hidden, chunk_segments)
    initial_conv = cache.layers[0].conv_states[0]
    initial_recurrent = cache.layers[0].recurrent_states[0]
    conv_token, conv_state_token = convolve(module, token['qkv'], initial_conv, mask, token_segments)
    conv_chunk_same, conv_state_chunk_same = convolve(module, token['qkv'], initial_conv, mask, chunk_segments)
    conv_chunk, conv_state_chunk = convolve(module, chunk['qkv'], initial_conv, mask, chunk_segments)
    rec_token, rec_state_token = recurrent(module, token, conv_token, initial_recurrent, mask, token_segments)
    rec_chunk_same, rec_state_chunk_same = recurrent(module, token, conv_token, initial_recurrent, mask, chunk_segments)
    rec_chunk, rec_state_chunk = recurrent(module, chunk, conv_chunk, initial_recurrent, mask, chunk_segments)
    result = {'config': config, 'task_ids': [task['task_id'] for task in tasks],
        'batch_size': len(tasks), 'segments': chunk_segments, 'valid_tokens': int(mask.sum()),
        'projection': {name: compare(chunk[name], token[name], mask) for name in token},
        'convolution_identical_inputs': compare(conv_chunk_same, conv_token, mask),
        'convolution_state_identical_inputs': difference(conv_state_chunk_same, conv_state_token),
        'convolution_batched_projections': compare(conv_chunk, conv_token, mask),
        'convolution_state_batched_projections': difference(conv_state_chunk, conv_state_token),
        'recurrence_identical_inputs': compare(rec_chunk_same, rec_token, mask),
        'recurrence_state_identical_inputs': difference(rec_state_chunk_same, rec_state_token),
        'recurrence_batched_projections': compare(rec_chunk, rec_token, mask),
        'recurrence_state_batched_projections': difference(rec_state_chunk, rec_state_token)}
    (args.output / 'comparison.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
