import argparse
import json
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace

import torch
from transformers.cache_utils import LinearAttentionCacheLayerMixin

from baselines.common.config import SharedConfig
from baselines.common.environment import TaskEnvironment
from baselines.common.tasks import read, rows
from baselines.latentmas.adapter import AGENTS, SOURCE, ModelAdapter
from baselines.latentmas.cache_compaction import compact_attention_history
from baselines.latentmas.common_service import LatentDecode, role_messages
from baselines.latentmas.native_hybrid_transport import CapturedPaddingTransport
from baselines.latentmas.numerics import difference
from jev_spawn.infra.backend import Backend
from jev_spawn.infra.prompts import load_prompt
from methods.evidence_flow.environment import EvidenceEnvironment


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def tensors(cache):
    result = []
    for layer in cache.layers:
        if isinstance(layer, LinearAttentionCacheLayerMixin):
            result.extend(layer.conv_states.values())
            result.extend(layer.recurrent_states.values())
        else:
            result.extend((layer.keys, layer.values, layer.cumulative_length))
    return result


def restore(destinations, saved):
    for destination, source in zip(destinations, saved, strict=True):
        destination.copy_(source)


def trajectory(decoder, config, shared, reference):
    snapshot = decoder.capture_snapshot()
    decoder.step()
    eager = decoder.logits.clone()
    decoder.restore_capture_snapshot(snapshot)
    started = time.perf_counter()
    decoder.capture(shared.runtime.graph_warmup_steps)
    capture_seconds = time.perf_counter() - started
    decoder.graph.replay()
    assert torch.equal(eager, decoder.logits)
    decoder.restore_capture_snapshot(snapshot)
    tokens, logits, positions = [], [], []
    events = [torch.cuda.Event(enable_timing=True) for _ in range(config['decode_steps'] + 1)]
    events[0].record()
    for step in range(config['decode_steps']):
        if reference is not None:
            decoder.ids.copy_(reference['inputs'][step])
        tokens.append(decoder.ids.clone())
        positions.append(decoder.positions.clone())
        decoder.graph.replay()
        logits.append(decoder.logits.clone())
        events[step + 1].record()
    torch.cuda.synchronize()
    record = {'capture_seconds': capture_seconds, 'eager_graph_exact': True,
        'per_step_ms': [left.elapsed_time(right) for left, right in zip(events, events[1:])],
        'output_tokens': torch.stack(logits).argmax(-1).tolist(),
        'position_ids': torch.cat(positions, dim=-1).tolist()}
    if reference is not None:
        record['comparison'] = difference(torch.stack(logits), torch.stack(reference['logits']))
        record['allclose'] = bool(torch.allclose(torch.stack(logits), torch.stack(reference['logits']),
                                                rtol=config['rtol'], atol=config['atol']))
        record['same_token_count'] = int((torch.stack(logits).argmax(-1) == torch.stack(reference['logits']).argmax(-1)).sum())
        assert all(torch.equal(left, right) for left, right in zip(positions, reference['positions'], strict=True))
    return record, {'inputs': tokens, 'logits': logits, 'positions': positions}


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    config = read(args.config)
    shared = SharedConfig.load(config['shared_config'])
    method, tools = read(config['method']), read(config['environment'])
    tasks = rows(config['tasks'])[config['task_start']:config['task_start'] + config['task_count']]
    assert len(tasks) == shared.runtime.batch_size == config['task_count']
    output = Path(config['output'])
    output.mkdir(parents=True, exist_ok=False)
    save(output / 'protocol.json', {'config': config, 'method': method,
        'upstream_revision': subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip(),
        'task_ids': [task['task_id'] for task in tasks], 'labels_loaded': False})
    backend = Backend(shared.backend())
    prompts = load_prompt(method['prompts'])
    evidence = EvidenceEnvironment(**tools['evidence'])
    messages = []
    for task in tasks:
        environment = TaskEnvironment(task, tools, output / 'tools' / task['task_id'],
                                      deadline=lambda: None, evidence=evidence)
        task_messages = [{'role': 'system', 'content': prompts['system']},
                         {'role': 'user', 'content': environment.reset()}]
        messages.append(role_messages(task_messages, prompts, shared.model.path))
    batches = []
    for role in range(len(AGENTS.default_agents())):
        rendered = [backend.tokenizer.apply_chat_template(item[role], tokenize=False,
                    add_generation_prompt=True, enable_thinking=False) for item in messages]
        batches.append(backend.tokenizer(rendered, padding=True, add_special_tokens=False,
                                        return_tensors='pt', truncation=False).to(backend.device))
    save(output / 'inputs.json', [{'input_ids': batch['input_ids'].tolist(),
        'attention_mask': batch['attention_mask'].tolist()} for batch in batches])
    width = sum(batch['input_ids'].shape[1] for batch in batches)
    width += (len(batches) - 1) * method['settings']['latent_steps']
    assert width <= shared.model.max_input_tokens
    decoder = LatentDecode(backend, len(tasks), width + shared.generation.max_new_tokens,
        graph_pool=torch.cuda.graph_pool_handle(), graph_stream=torch.cuda.Stream(device=backend.device))
    wrapper = ModelAdapter(backend, SimpleNamespace(latent_space_realign=True))
    alignment = wrapper._ensure_latent_realign_matrix(wrapper.model, backend.device, wrapper.args)
    deadline = time.perf_counter() + config['validation_timeout_seconds']

    def check_deadline():
        assert time.perf_counter() < deadline, 'Cache-compaction qualification exceeded declared time.'

    transport = CapturedPaddingTransport(backend, decoder, shared.runtime.graph_warmup_steps,
        method['settings']['padding_workspace_reserve_bytes'], check_deadline)
    wrapper.model = transport
    wrapper._latent_realign_matrices = {id(transport): alignment}
    for agent, batch in zip(AGENTS.default_agents(), batches, strict=True):
        started = time.perf_counter()
        transport.phase = agent.role
        if agent.role != 'judger':
            wrapper.generate_latent_batch(**batch, latent_steps=method['settings']['latent_steps'],
                                          past_key_values=transport.cache)
        else:
            transport(**batch, past_key_values=transport.cache)
        torch.cuda.synchronize()
        print(json.dumps({'role': agent.role, 'seconds': time.perf_counter() - started,
                          'physical_width': transport.mask.shape[1]}), flush=True)
        save(output / 'forwards.json', transport.records)
    assert transport.mask.shape[1] == width
    original_mask = transport.mask.clone()
    destinations = tensors(decoder.cache)
    saved = [tensor.clone() for tensor in destinations]
    decoder.key_valid.fill_(True)
    decoder.key_valid[:, :width].copy_(original_mask)
    decoder.positions.copy_(original_mask.sum(-1)[:, None])
    decoder.logits.copy_(backend.model.lm_head(transport.last_hidden))
    decoder.ids.copy_(decoder.logits.argmax(-1)[:, None])
    pending_ids, pending_logits, pending_positions = decoder.ids.clone(), decoder.logits.clone(), decoder.positions.clone()
    grouped_kwargs = decoder.decode_kwargs
    decoder.decode_kwargs = {}
    original, reference = trajectory(decoder, config, shared, None)
    save(output / 'original_holes_native_attention.json', original)
    restore(destinations, saved)
    decoder.ids.copy_(pending_ids)
    decoder.logits.copy_(pending_logits)
    decoder.positions.copy_(pending_positions)
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    begin.record()
    compact_mask = compact_attention_history(decoder.cache, original_mask)
    end.record()
    torch.cuda.synchronize()
    compaction_ms = begin.elapsed_time(end)
    index = 0
    for layer in decoder.cache.layers:
        if isinstance(layer, LinearAttentionCacheLayerMixin):
            count = len(layer.conv_states) + len(layer.recurrent_states)
            assert all(torch.equal(destinations[j], saved[j]) for j in range(index, index + count))
            index += count
        else:
            for tensor in (layer.keys, layer.values):
                for row in range(len(tasks)):
                    assert torch.equal(tensor[row, :, :width][:, compact_mask[row].bool()],
                                       saved[index][row, :, :width][:, original_mask[row].bool()])
                index += 1
            assert torch.equal(layer.cumulative_length, saved[index])
            index += 1
    decoder.key_valid[:, :width].copy_(compact_mask)
    save(output / 'invariants.json', {'valid_kv_order_bitwise_preserved': True,
        'recurrent_states_bitwise_preserved': True, 'physical_lengths_bitwise_preserved': True,
        'physical_history_width': width, 'valid_history_tokens': original_mask.sum(-1).tolist(),
        'compaction_ms': compaction_ms})
    restore(saved, destinations)
    compacted, _ = trajectory(decoder, config, shared, reference)
    save(output / 'compacted_native_attention.json', compacted)
    restore(destinations, saved)
    decoder.ids.copy_(pending_ids)
    decoder.logits.copy_(pending_logits)
    decoder.positions.copy_(pending_positions)
    decoder.decode_kwargs = grouped_kwargs
    decoder.validate_attention()
    grouped, _ = trajectory(decoder, config, shared, reference)
    save(output / 'compacted_grouped_attention.json', grouped)
    save(output / 'summary.json', {'batch_size': len(tasks), 'task_ids': [task['task_id'] for task in tasks],
        'physical_history_width': width, 'valid_history_tokens': original_mask.sum(-1).tolist(),
        'original_mask': original_mask.tolist(), 'compacted_mask': compact_mask.tolist(),
        'valid_kv_order_bitwise_preserved': True, 'recurrent_states_bitwise_preserved': True,
        'physical_lengths_bitwise_preserved': True, 'pending_tokens_bitwise_preserved': True,
        'compaction_ms': compaction_ms, 'original': original, 'compacted': compacted, 'grouped': grouped,
        'decode_slots': config['decode_steps'] * len(tasks), 'production_promoted': False})


if __name__ == '__main__':
    main()
