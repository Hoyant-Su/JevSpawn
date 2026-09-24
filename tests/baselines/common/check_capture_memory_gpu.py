import argparse
import gc
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers.cache_utils import LinearAttentionCacheLayerMixin

from baselines.common.config import SharedConfig
from baselines.common.tasks import read
from baselines.latentmas.adapter import ModelAdapter
from baselines.latentmas.common_service import LatentDecode, LatentTransport, role_messages
from baselines.latentmas.native_hybrid_transport import CapturedPadding
from jev_spawn.infra.backend import Backend
from jev_spawn.runtime.cache_arena import StaticCacheArena
from jev_spawn.runtime.decoding import CapturedDecode


def state(decoder):
    tensors = [decoder.ids, decoder.positions, decoder.logits]
    for layer in decoder.cache.layers:
        if isinstance(layer, LinearAttentionCacheLayerMixin):
            tensors.extend(layer.conv_states.values())
            tensors.extend(layer.recurrent_states.values())
        else:
            tensors.extend([layer.keys, layer.values, layer.cumulative_length])
    return tensors


def exact(tensors, expected):
    checks = [torch.equal(a, b) for a, b in zip(tensors, expected, strict=True)]
    assert all(checks), f'Full-state equality failed at tensors {[i for i, equal in enumerate(checks) if not equal]}'
    return len(checks)


def clone_state(decoder, reserve):
    tensors = state(decoder)
    needed = sum(t.numel() * t.element_size() for t in tensors)
    free, _ = torch.cuda.mem_get_info(decoder.backend.device)
    available = free + torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
    assert available >= needed + reserve, f'GPU-only parity snapshot needs {needed + reserve} bytes; have {available}'
    assert all(t.device.type == 'cuda' for t in tensors)
    return [t.clone() for t in tensors]


def check(decoder, warmup, reserve):
    before = clone_state(decoder, reserve)
    if decoder.graph is None:
        decoder.capture(warmup)
    restored = exact(state(decoder), before)
    del before
    snapshot = decoder.capture_snapshot()
    decoder.step()
    expected = clone_state(decoder, reserve)
    hidden = decoder.hidden.clone() if isinstance(decoder, CapturedPadding) else None
    decoder.restore_capture_snapshot(snapshot)
    del snapshot
    decoder.graph.replay()
    torch.cuda.synchronize()
    compared = exact(state(decoder), expected)
    if hidden is not None:
        exact([decoder.hidden], [hidden])
    del expected
    return {'restored_tensors': restored, 'eager_graph_equal_tensors': compared,
            'capture_memory': decoder.capture_memory,
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
            'peak_reserved_bytes': torch.cuda.max_memory_reserved()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config = read(args.config)
    shared = SharedConfig.load(config['shared_config'])
    args.output.mkdir(parents=True, exist_ok=False)
    source = read(config['source_batches'])[config['source_batch_index']]
    assert source['batch_size'] == shared.runtime.batch_size
    backend = Backend(shared.backend())
    wrapper = ModelAdapter(backend, SimpleNamespace(latent_space_realign=True))
    alignment = wrapper._ensure_latent_realign_matrix(wrapper.model, backend.device, wrapper.args)
    capacity = shared.model.max_input_tokens + shared.generation.max_new_tokens
    assert capacity == config['required_capacity']
    arena = StaticCacheArena(backend.model.config, shared.runtime.batch_size, capacity,
                             backend.model.lm_head.weight.dtype, backend.device)
    pool, stream = torch.cuda.graph_pool_handle(), torch.cuda.Stream(device=backend.device)
    rendered = [backend.tokenizer.apply_chat_template(messages, tokenize=False,
                add_generation_prompt=True, enable_thinking=False) for messages in source['messages']]
    inputs = backend.tokenizer(rendered, padding=True, add_special_tokens=False,
                               truncation=False, return_tensors='pt').to(backend.device)
    assert inputs['attention_mask'].sum(1).tolist() == source['input_tokens']
    assert inputs['input_ids'].shape[1] <= shared.model.max_input_tokens
    report = {'config': config, 'task_ids': source['task_ids'], 'input_tokens': source['input_tokens'],
              'capacity': capacity, 'arena_bytes': arena.nbytes, 'snapshots_gpu_only': True,
              'diagnostic_memory': 'Peak counters include one full-state GPU reference of full_state_bytes; production capture rollback uses saved_state_bytes. No diagnostic KV clone exists during prefill.',
              'comparisons_bit_exact': True, 'checks': [], 'passed': False}
    save = lambda: (args.output / 'comparison.json').write_text(json.dumps(report, indent=2) + '\n')
    save()
    decoders = {}
    with torch.inference_mode():
        for size in config['replay_batch_order']:
            if size not in decoders:
                decoders[size] = CapturedDecode(backend, size, capacity, arena,
                                                graph_pool=pool, graph_stream=stream)
            decoder = decoders[size]
            persistent = {n: [t.clone() for t in (d.ids, d.positions, d.logits)]
                          for n, d in decoders.items() if n != size}
            selected = {name: tensor[:size] for name, tensor in inputs.items()}
            torch.cuda.reset_peak_memory_stats()
            decoder.prefill(selected)
            prefill_peak = torch.cuda.max_memory_allocated()
            proof = check(decoder, shared.runtime.graph_warmup_steps, config['snapshot_reserve_bytes'])
            for n, tensors in persistent.items():
                exact([decoders[n].ids, decoders[n].positions, decoders[n].logits], tensors)
            report['checks'].append({'phase': 'text', 'batch_size': size,
                                    'prefill_peak_allocated_bytes': prefill_peak, **proof})
            save()
        method = read(config['method'])
        prompts = read(method['prompts'])
        # Use original qualification messages for the latent algorithm's full role history.
        original = read(config['latent_source_batches'])[config['latent_source_batch_index']]
        messages = [role_messages(m, prompts, shared.model.path) for m in original['messages']]
        encoded = []
        for role in range(config['latent_roles_to_encode']):
            text = [backend.tokenizer.apply_chat_template(m[role], tokenize=False,
                    add_generation_prompt=True, enable_thinking=False) for m in messages]
            encoded.append(backend.tokenizer(text, padding=True, add_special_tokens=False,
                                            truncation=False, return_tensors='pt').to(backend.device))
        decoder = LatentDecode(backend, shared.runtime.batch_size, capacity, arena,
                                graph_pool=pool, graph_stream=stream)
        decoder.cache.reset()
        transport = LatentTransport(backend, decoder.cache, lambda: None)
        wrapper.model = transport
        wrapper._latent_realign_matrices = {id(transport): alignment}
        wrapper.generate_latent_batch(**encoded[0], latent_steps=method['settings']['latent_steps'],
                                      past_key_values=None)
        critic = encoded[1]
        graph = CapturedPadding(backend, decoder.cache, shared.runtime.batch_size, capacity,
            method['settings']['padding_workspace_reserve_bytes'], graph_pool=pool, graph_stream=stream)
        graph.ids.zero_()
        graph.logits.zero_()
        history = torch.cat([transport.mask, critic['attention_mask']], dim=1)
        graph.key_valid.fill_(True)
        graph.key_valid[:, :history.shape[1]].copy_(history)
        positions = (history.cumsum(1) - 1).clamp_min(0)[:, -critic['input_ids'].shape[1]:]
        embeddings = backend.model.get_input_embeddings()(critic['input_ids'])
        persistent = {n: [t.clone() for t in (d.ids, d.positions, d.logits)] for n, d in decoders.items()}
        assert config['padding_tokens'] <= critic['input_ids'].shape[1]
        for offset in range(config['padding_tokens']):
            graph.embeddings.copy_(embeddings[:, offset:offset + 1])
            graph.positions.copy_(positions[:, offset:offset + 1])
            graph.active.copy_(critic['attention_mask'][:, offset].bool())
            if offset in config['padding_check_offsets']:
                proof = check(graph, shared.runtime.graph_warmup_steps, config['snapshot_reserve_bytes'])
                report['checks'].append({'phase': 'latent_padding', 'offset': offset, **proof})
                save()
            else:
                graph.graph.replay()
        report['padding_tokens_per_row'] = config['padding_tokens']
        report['padding_valid_tokens_per_row'] = critic['attention_mask'][:, :config['padding_tokens']].sum(1).tolist()
        for n, tensors in persistent.items():
            exact([decoders[n].ids, decoders[n].positions, decoders[n].logits], tensors)
        padding_outputs = [t.clone() for t in (graph.ids, graph.positions, graph.logits, graph.hidden)]
        transport.mask = history[:, :transport.mask.shape[1] + config['padding_tokens']]
        transport.last_hidden = graph.hidden
        decoder.initialize(transport)
        for step in range(config['latent_text_replays']):
            proof = check(decoder, shared.runtime.graph_warmup_steps, config['snapshot_reserve_bytes'])
            exact([graph.ids, graph.positions, graph.logits, graph.hidden], padding_outputs)
            report['checks'].append({'phase': 'text_from_real_latent_cache', 'step': step, **proof})
            save()
        report['graph_pool_ids'] = [list(d.graph.pool()) for d in decoders.values()] + [list(graph.graph.pool()), list(decoder.graph.pool())]
        assert all(tuple(pool_id) == pool for pool_id in report['graph_pool_ids'])
        report['passed'] = True
        save()
    del decoders, decoder, graph, wrapper, transport, arena
    gc.collect()


if __name__ == '__main__':
    main()
