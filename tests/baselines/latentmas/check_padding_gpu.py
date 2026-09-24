import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
from types import SimpleNamespace

import torch
from transformers import AutoTokenizer, StaticCache
from transformers.cache_utils import LinearAttentionCacheLayerMixin

from baselines.common.config import SharedConfig
from baselines.common.environment import TaskEnvironment
from baselines.common.tasks import read, rows
from baselines.latentmas.adapter import HybridTransport, ModelAdapter
from baselines.latentmas.common_service import LatentDecode, role_messages
from baselines.latentmas.cached_prefix import CachedPrefixTransport
from baselines.latentmas.native_hybrid_transport import CapturedPaddingTransport, padding_segments
from baselines.latentmas.recurrent_segments import RecurrentSegmentTransport
from baselines.latentmas.numerics import difference
from jev_spawn.infra.backend import Backend
from jev_spawn.infra.configuration import CORE
from jev_spawn.infra.prompts import load_prompt
from jev_spawn.runtime.cache_arena import StaticCacheArena
from methods.evidence_flow.environment import EvidenceEnvironment


def save(path, value):
    temporary = path.with_suffix('.partial.json')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


class StaticReferenceTransport(HybridTransport):
    def __init__(self, backend, cache):
        super().__init__(backend)
        self.initial_cache = cache

    def _forward(self, embeddings, mask, cache):
        return super()._forward(embeddings, mask, self.initial_cache if cache is None else cache)


def snapshot_tensors(transport):
    tensors = [transport.mask, transport.last_hidden]
    for layer in transport.cache.layers:
        if isinstance(layer, LinearAttentionCacheLayerMixin):
            tensors.extend([layer.conv_states[0], layer.recurrent_states[0]])
        else:
            tensors.extend([layer.keys, layer.values, layer.cumulative_length])
    assert all(tensor.device.type == 'cuda' for tensor in tensors)
    return tensors


def memory_available(device):
    free, total = torch.cuda.mem_get_info(device)
    reusable = torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
    return {'device_free_bytes': free, 'allocator_reusable_bytes': reusable,
            'available_bytes': free + reusable, 'device_total_bytes': total}


def snapshot(transport, *, clone, workspace_reserve):
    tensors = snapshot_tensors(transport)
    required = sum(tensor.numel() * tensor.element_size() for tensor in tensors)
    if clone:
        available = memory_available(transport.backend.device)['available_bytes']
        assert available >= required + workspace_reserve, (
            f'Insufficient GPU memory for exact cache snapshot: need {required} snapshot bytes '
            f'plus {workspace_reserve} workspace bytes, have {available}. CPU offload is forbidden.')
    copy = lambda tensor: tensor.detach().clone() if clone else tensor.detach()
    state = {'mask': copy(transport.mask), 'hidden': copy(transport.last_hidden), 'layers': [],
             'tensor_bytes': required, 'cloned': clone,
             'physical_length': int(transport.cache.get_seq_length()),
             'cache_classes': [type(layer).__name__ for layer in transport.cache.layers]}
    for layer in transport.cache.layers:
        if isinstance(layer, LinearAttentionCacheLayerMixin):
            state['layers'].append({'conv': copy(layer.conv_states[0]),
                'recurrent': copy(layer.recurrent_states[0]), 'has_previous_state': dict(layer.has_previous_state)})
        else:
            state['layers'].append({'keys': copy(layer.keys), 'values': copy(layer.values),
                                   'cumulative_length': copy(layer.cumulative_length)})
    return state


def restore(transport, cache, state):
    for layer, record in zip(cache.layers, state['layers']):
        if isinstance(layer, LinearAttentionCacheLayerMixin):
            layer.conv_states[0].copy_(record['conv'])
            layer.recurrent_states[0].copy_(record['recurrent'])
            layer.has_previous_state = dict(record['has_previous_state'])
            assert torch.equal(layer.conv_states[0], record['conv'])
            assert torch.equal(layer.recurrent_states[0], record['recurrent'])
        else:
            layer.keys.copy_(record['keys'])
            layer.values.copy_(record['values'])
            layer.cumulative_length.copy_(record['cumulative_length'])
            assert torch.equal(layer.keys, record['keys'])
            assert torch.equal(layer.values, record['values'])
    transport.mask = state['mask']
    transport.last_hidden = state['hidden']
    transport.cache = cache
    transport.phase = 'critic_prefix_validation'
    assert int(cache.get_seq_length()) == state['physical_length']


def compare_tensor(candidate, reference, tolerance):
    assert candidate.shape == reference.shape
    assert bool(torch.isfinite(candidate).all()) and bool(torch.isfinite(reference).all())
    close = torch.isclose(candidate.float(), reference.float(), **tolerance)
    return {**difference(candidate, reference), 'outside_tolerance': int((~close).sum()),
            'passed': bool(close.all())}


def run_prefix(transport, embeddings, mask, segments, deadline, *, clone_state, workspace_reserve, execution):
    outputs, positions = [], []

    def observe(module, args, kwargs):
        positions.append(kwargs['position_ids'].detach().clone())

    started = time.perf_counter()
    if execution in ('eager', 'fused_recurrent_segments', 'cached_update_segments'):
        hook = transport.trunk.register_forward_pre_hook(observe, with_kwargs=True)
        try:
            for start, end in segments:
                assert time.perf_counter() < deadline, 'Declared validation deadline exceeded.'
                result = transport._forward(embeddings[:, start:end], mask[:, start:end], transport.cache)
                outputs.append(result.last_hidden_state.detach().clone())
                transport.records[-1].update(segment_start=start, segment_end=end)
        finally:
            hook.remove()
    else:
        assert execution == 'cuda_graph_single_token'

        def record(hidden, position):
            outputs.append(hidden[:, None].detach().clone())
            positions.append(position.detach().clone())

        transport.run_padding(embeddings, mask, observer=record)
    torch.cuda.synchronize(transport.backend.device)
    return {'hidden': torch.cat(outputs, dim=1), 'positions': torch.cat(positions, dim=1),
            'state': snapshot(transport, clone=clone_state, workspace_reserve=workspace_reserve), 'forwards': transport.records,
            'elapsed_seconds': time.perf_counter() - started}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config = read(args.config)
    shared = SharedConfig.load(config['shared_config'])
    method, tools = read(config['method']), read(config['environment'])
    prompts = load_prompt(method['prompts'])
    all_tasks = rows(config['tasks'])
    tasks = all_tasks[config['task_start']:config['task_start'] + config['task_count']]
    assert config['task_start'] == 0
    assert config['candidate_transport'] in ('cuda_graph_single_token', 'fused_recurrent_segments', 'cached_update_segments')
    assert len(tasks) == config['task_count'] == shared.runtime.batch_size
    assert method['settings']['latent_steps'] == 10
    assert shared.model.dtype == 'bfloat16' and shared.model.kernel == 'fla'
    args.output.mkdir(parents=True, exist_ok=False)
    save(args.output / 'protocol.json', {'config': config, 'shared': asdict(shared),
        'method': method, 'prompts': prompts, 'task_ids': [task['task_id'] for task in tasks], 'labels_loaded': False})
    tokenizer = AutoTokenizer.from_pretrained(shared.model.path,
        padding_side=CORE['backend']['padding_side'], local_files_only=CORE['backend']['local_files_only'])
    evidence = EvidenceEnvironment(**tools['evidence'])
    role_batches = []
    for task in tasks:
        environment = TaskEnvironment(task, tools, args.output / 'tools' / task['task_id'],
                                      deadline=lambda: None, evidence=evidence)
        messages = [{'role': 'system', 'content': prompts['system']},
                    {'role': 'user', 'content': environment.reset()}]
        role_batches.append(role_messages(messages, prompts, shared.model.path))
    encoded = []
    for role in range(2):
        rendered = [tokenizer.apply_chat_template(batch[role], tokenize=False,
                    add_generation_prompt=True, enable_thinking=False) for batch in role_batches]
        encoded.append(tokenizer(rendered, padding=True, add_special_tokens=False,
                                 return_tensors='pt', truncation=False))
    planner, critic = encoded
    prefix = config['critic_prefix_tokens']
    full_segments = padding_segments(critic['attention_mask'])
    segments = [(start, min(end, prefix)) for start, end in full_segments if start < prefix]
    prefix_mask = critic['attention_mask'][:, :prefix]
    save(args.output / 'padding_layout.json', {
        'task_ids': [task['task_id'] for task in tasks],
        'planner_shape': list(planner['input_ids'].shape),
        'critic_shape': list(critic['input_ids'].shape),
        'critic_left_padding': (critic['attention_mask'] == 0).sum(1).tolist(),
        'critic_segments': full_segments, 'selected_segments': segments,
        'prefix_rows_with_valid_tokens': int(prefix_mask.bool().any(1).sum())})
    assert 0 < prefix <= critic['input_ids'].shape[1]
    assert len(segments) - 1 >= config['minimum_internal_padding_boundaries']
    assert int(prefix_mask.bool().any(1).sum()) >= config['minimum_rows_with_valid_prefix_tokens']
    backend = Backend(shared.backend())
    planner, critic = (batch.to(backend.device) for batch in encoded)
    prefix_mask = critic['attention_mask'][:, :prefix]
    required = planner['input_ids'].shape[1] + method['settings']['latent_steps'] + prefix
    block = shared.runtime.graph_cache_block_tokens
    capacity = (required + block - 1) // block * block
    cache = StaticCache(config=backend.model.config, max_cache_len=capacity)
    reference = StaticReferenceTransport(backend, cache)
    wrapper = ModelAdapter(backend, SimpleNamespace(latent_space_realign=True))
    wrapper.model = reference
    with torch.inference_mode():
        wrapper._ensure_latent_realign_matrix(reference, backend.device, wrapper.args)
        torch.cuda.synchronize(backend.device)
        started = time.perf_counter()
        deadline = started + config['validation_timeout_seconds']
        wrapper.generate_latent_batch(**planner, latent_steps=method['settings']['latent_steps'], past_key_values=None)
        cache_bytes = sum(tensor.numel() * tensor.element_size() for tensor in snapshot_tensors(reference))
        hidden_size = reference.last_hidden.shape[-1]
        hidden_bytes = len(tasks) * prefix * hidden_size * reference.last_hidden.element_size()
        position_bytes = len(tasks) * prefix * critic['input_ids'].element_size()
        prefix_storage = 5 * hidden_bytes + 4 * position_bytes
        memory = memory_available(backend.device)
        final_cache_bytes = cache_bytes + prefix_mask.numel() * prefix_mask.element_size()
        required_extra = cache_bytes + final_cache_bytes + prefix_storage + config['workspace_reserve_bytes']
        save(args.output / 'memory.json', {**memory, 'single_cache_snapshot_bytes': cache_bytes,
             'final_cache_snapshot_bytes': final_cache_bytes,
             'cache_clone_count': 2, 'candidate_cache_is_view': True,
             'prefix_buffer_reservation_bytes': prefix_storage,
             'workspace_reserve_bytes': config['workspace_reserve_bytes'],
             'required_extra_bytes': required_extra, 'host_tensor_offload': False})
        assert memory['available_bytes'] >= required_extra, (
            f'Insufficient GPU memory: need {required_extra} additional bytes, '
            f"have {memory['available_bytes']}. No CPU offload or device bouncing is allowed.")
        initial = snapshot(reference, clone=True, workspace_reserve=config['workspace_reserve_bytes'])
        save(args.output / 'inputs.json', {'planner_valid_tokens_per_row': planner['attention_mask'].sum(1).tolist(),
             'critic_full_padded_width': critic['input_ids'].shape[1], 'critic_prefix_tokens': prefix,
             'compared_valid_prefix_tokens': int(prefix_mask.sum()),
             'valid_prefix_tokens_per_row': prefix_mask.sum(1).tolist(), 'segments': segments,
             'cache_capacity': capacity, 'planner_forward_calls': reference.records,
             'planner_latent_steps': method['settings']['latent_steps'], 'initial_cache_physical_length': initial['physical_length']})
        embeddings = backend.model.get_input_embeddings()(critic['input_ids'][:, :prefix])
        reference.records = []
        reference.phase = 'critic_prefix_validation'
        old = run_prefix(reference, embeddings, prefix_mask, [(i, i + 1) for i in range(prefix)], deadline,
                         clone_state=True, workspace_reserve=config['workspace_reserve_bytes'], execution='eager')
        save(args.output / 'reference_forwards.json', {'seconds': old['elapsed_seconds'], 'forwards': old['forwards']})
        def check_deadline():
            assert time.perf_counter() < deadline, 'Declared validation deadline exceeded.'

        arena_proof = None
        if shared.runtime.cache_allocation == 'shared_static_cache_arena_v1':
            alignment = wrapper._latent_realign_matrices[id(reference)]
            wrapper.model = None
            del reference, cache
            arena_capacity = (shared.model.max_input_tokens + shared.generation.max_new_tokens + block - 1) // block * block
            arena = StaticCacheArena(backend.model.config, shared.runtime.batch_size, arena_capacity,
                                    backend.model.lm_head.weight.dtype, backend.device)
            decoder = LatentDecode(backend, len(tasks), capacity, arena=arena)
            cache = decoder.cache
            for layer, storage in zip(cache.layers, arena.storage, strict=True):
                pairs = [(layer.conv_states[0], storage['conv']),
                         (layer.recurrent_states[0], storage['recurrent'])] if isinstance(layer, LinearAttentionCacheLayerMixin) else [
                         (layer.keys, storage['keys']), (layer.values, storage['values'])]
                for view, owner in pairs:
                    assert view.untyped_storage().data_ptr() == owner.untyped_storage().data_ptr()
            cache.reset()
            planner_candidate = StaticReferenceTransport(backend, cache)
            wrapper.model = planner_candidate
            wrapper._latent_realign_matrices = {id(planner_candidate): alignment}
            wrapper.generate_latent_batch(**planner, latent_steps=method['settings']['latent_steps'], past_key_values=None)
            planner_state = snapshot(planner_candidate, clone=False, workspace_reserve=config['workspace_reserve_bytes'])
            assert torch.equal(planner_state['hidden'], initial['hidden'])
            assert torch.equal(planner_state['mask'], initial['mask'])
            assert planner_state['physical_length'] == initial['physical_length']
            for actual, expected in zip(planner_state['layers'], initial['layers'], strict=True):
                names = ('conv', 'recurrent') if 'conv' in actual else ('keys', 'values', 'cumulative_length')
                for name in names:
                    assert torch.equal(actual[name], expected[name]), 'Arena planner state differs: ' + name
            arena_proof = {'planner_and_ten_latent_steps_bit_exact': True,
                'arena_bytes': arena.nbytes, 'arena_max_cache_tokens': arena_capacity,
                'decoder_cache_bound_directly_to_arena': True, 'independent_reference_cache_released': True}
            save(args.output / 'arena_planner.json', arena_proof)
            del planner_state, planner_candidate
        else:
            assert shared.runtime.cache_allocation == 'independent_static_cache_v1'
            decoder = SimpleNamespace(cache=cache, capacity=capacity, padding_graph=None)
        candidates = {
            'cuda_graph_single_token': lambda: CapturedPaddingTransport(
                backend, decoder, shared.runtime.graph_warmup_steps,
                config['graph_workspace_reserve_bytes'], check_deadline),
            'fused_recurrent_segments': lambda: RecurrentSegmentTransport(backend),
            'cached_update_segments': lambda: CachedPrefixTransport(backend),
        }
        candidate = candidates[config['candidate_transport']]()
        restore(candidate, cache, initial)
        inactive = ~prefix_mask.bool().any(1)
        for previous, current in zip(initial['layers'], old['state']['layers']):
            if 'conv' in current:
                assert torch.equal(current['conv'][inactive], previous['conv'][inactive])
                assert torch.equal(current['recurrent'][inactive], previous['recurrent'][inactive])
        del initial
        new = run_prefix(candidate, embeddings, prefix_mask, segments, deadline,
                         clone_state=False, workspace_reserve=config['workspace_reserve_bytes'],
                         execution=config['candidate_transport'])
        if config['candidate_transport'] == 'cuda_graph_single_token':
            save(args.output / 'graph_memory.json', decoder.padding_graph.memory)
        save(args.output / 'candidate_forwards.json', {'seconds': new['elapsed_seconds'], 'forwards': new['forwards']})
    assert torch.equal(old['positions'], new['positions'])
    assert torch.equal(old['state']['mask'], new['state']['mask'])
    assert old['state']['physical_length'] == new['state']['physical_length'] == required
    mask = old['state']['mask'].bool()
    inactive = ~prefix_mask.bool().any(1)
    for previous, current in zip(old['state']['layers'], new['state']['layers']):
        if 'conv' in current:
            assert torch.equal(current['conv'][inactive], previous['conv'][inactive])
            assert torch.equal(current['recurrent'][inactive], previous['recurrent'][inactive])
    compared = [{'tensor': 'all_valid_hidden_outputs', **compare_tensor(
        new['hidden'][prefix_mask.bool()], old['hidden'][prefix_mask.bool()], config['tolerance'])}]
    for index, (left, right) in enumerate(zip(new['state']['layers'], old['state']['layers'])):
        for name in ('conv', 'recurrent') if 'conv' in left else ('keys', 'values'):
            for row in range(len(tasks)):
                candidate_tensor, reference_tensor = left[name][row], right[name][row]
                if name in ('keys', 'values'):
                    candidate_tensor = candidate_tensor[:, :required][:, mask[row]]
                    reference_tensor = reference_tensor[:, :required][:, mask[row]]
                compared.append({'layer': index, 'row': row, 'tensor': name,
                                 **compare_tensor(candidate_tensor, reference_tensor, config['tolerance'])})
    passed = all(record['passed'] for record in compared)
    summary = {'scope': config['scope'], 'passed': passed, 'tolerance': config['tolerance'],
        'batch_size': len(tasks), 'latent_steps': method['settings']['latent_steps'],
        'critic_prefix_tokens': prefix, 'compared_valid_prefix_tokens': int(prefix_mask.sum()),
        'candidate_transport': config['candidate_transport'],
        'candidate_cache_allocation': shared.runtime.cache_allocation, 'arena_proof': arena_proof,
        'snapshot_storage': 'GPU only; initial snapshot released before graph capture; candidate cache views',
        'host_tensor_offload': False, 'exact_initial_cache_restore': True, 'inactive_states_unchanged': True,
        'positions_equal': True, 'masks_equal': True,
        'physical_cache_length': required, 'reference_forwards': len(old['forwards']),
        'candidate_forwards': len(new['forwards']), 'reference_seconds': old['elapsed_seconds'],
        'candidate_seconds': new['elapsed_seconds'], 'validation_elapsed_seconds': time.perf_counter() - started,
        'failed_tensors': sum(not record['passed'] for record in compared), 'comparisons': compared}
    save(args.output / 'comparison.json', summary)
    print(json.dumps({key: value for key, value in summary.items() if key != 'comparisons'}), flush=True)
    assert time.perf_counter() < deadline, 'Declared validation deadline exceeded.'
    assert passed, 'BF16 cache transport comparison exceeded the predeclared numerical tolerance; see comparison.json.'


if __name__ == '__main__':
    main()
