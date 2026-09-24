import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import statistics
import time
from types import SimpleNamespace

import torch
import torch.distributed as dist

from baselines.common.config import SharedConfig
from baselines.common.tasks import read
from baselines.latentmas.adapter import AGENTS, ModelAdapter, OriginalWrapper
from baselines.latentmas.common_service import LatentDecode, SegmentedLatentTransport
from baselines.latentmas.numerics import difference
from jev_spawn.infra.backend import Backend
from jev_spawn.infra.qwen35.commands import ParallelCommands


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


class ObservedAdapter(ModelAdapter):
    def _apply_latent_realignment(self, hidden, model):
        actual = super()._apply_latent_realignment(hidden, model)
        expected = self.reference._apply_latent_realignment(hidden, model)
        self.hidden.append(hidden.clone())
        self.aligned.append(actual.clone())
        self.reference_aligned.append(expected.clone())
        return actual


def native_alignment(wrapper, backend):
    head = backend.model.lm_head
    full = torch.empty((head.vocab_size, head.hidden_size), dtype=head.weight.dtype, device=backend.device)
    dist.all_gather_into_tensor(full, head.weight, group=head.group)
    reference_model = SimpleNamespace(get_input_embeddings=backend.model.get_input_embeddings,
        get_output_embeddings=lambda: SimpleNamespace(weight=full))
    return OriginalWrapper._build_latent_realign_matrix(wrapper, reference_model, backend.device, wrapper.args)


@torch.inference_mode()
def run(settings, commands):
    shared = SharedConfig.load(settings['shared_config'])
    assert dist.get_world_size() == shared.runtime.world_size
    path = Path(settings['output'])
    if commands.is_leader:
        path.mkdir(parents=True, exist_ok=False)
        save(path / 'protocol.json', settings)
    dist.barrier()
    backend = Backend(shared.backend())
    backend.parallel_commands = commands
    method = read(settings['method'])
    args = SimpleNamespace(latent_space_realign=method['settings']['alignment'],
                           alignment_settings=read(method['settings']['alignment_config']))
    wrapper, reference = ObservedAdapter(backend, args), ModelAdapter(backend, args)
    actual_alignment = wrapper._ensure_latent_realign_matrix(wrapper.model, backend.device, args)
    expected_alignment = native_alignment(wrapper, backend)
    alignment_record = {'matrix': difference(actual_alignment[0], expected_alignment[0]),
        'target_norm_bitwise_equal': torch.equal(actual_alignment[1], expected_alignment[1]),
        'matrix_allclose': torch.allclose(actual_alignment[0], expected_alignment[0],
                                          rtol=settings['rtol'], atol=settings['atol']),
        'local_output_rows': backend.model.lm_head.weight.shape[0], 'global_output_rows': backend.vocab_size,
        'row_start': dist.get_rank() * backend.model.lm_head.weight.shape[0]}
    assert alignment_record['target_norm_bitwise_equal'] and alignment_record['matrix_allclose']
    if commands.is_leader:
        save(path / 'alignment.json', alignment_record)
    source = read(settings['source'])[settings['source_batch']]
    assert source['batch_size'] == shared.runtime.batch_size
    batches = []
    for role in range(len(AGENTS.default_agents())):
        text = [wrapper.render_chat(messages[role]) for messages in source['role_messages']]
        batch = backend.tokenizer(text, padding=True, truncation=False, add_special_tokens=False,
                                  return_tensors='pt').to(backend.device)
        assert batch['attention_mask'].sum(-1).tolist() == [lengths[role] for lengths in source['role_input_tokens']]
        batches.append(batch)
    width = sum(batch['input_ids'].shape[1] for batch in batches)
    width += (len(batches) - 1) * method['settings']['latent_steps']
    assert width <= shared.model.max_input_tokens
    decoder = LatentDecode(backend, source['batch_size'], width + shared.generation.max_new_tokens,
                           graph_pool=torch.cuda.graph_pool_handle(), graph_stream=torch.cuda.Stream(device=backend.device))
    deadline = time.perf_counter() + settings['validation_timeout_seconds']

    def check_deadline():
        valid = commands.leader_value(time.perf_counter() < deadline if commands.is_leader else None)
        assert valid, 'Parallel alignment qualification exceeded its declared timeout.'

    transport = SegmentedLatentTransport(backend, decoder.cache, check_deadline)
    wrapper.model, wrapper.reference = transport, reference
    wrapper._latent_realign_matrices = {id(transport): actual_alignment}
    reference._latent_realign_matrices = {id(transport): expected_alignment}
    wrapper.hidden, wrapper.aligned, wrapper.reference_aligned = [], [], []
    roles = []
    for agent, batch in zip(AGENTS.default_agents(), batches, strict=True):
        started = time.perf_counter()
        if agent.role != 'judger':
            wrapper.generate_latent_batch(**batch, latent_steps=method['settings']['latent_steps'],
                                          past_key_values=transport.cache)
        else:
            transport(**batch, past_key_values=transport.cache)
        torch.cuda.synchronize()
        record = {'role': agent.role, 'seconds': time.perf_counter() - started,
                  'physical_width': transport.mask.shape[1], 'valid_lengths': transport.mask.sum(-1).tolist()}
        roles.append(record)
        if commands.is_leader:
            save(path / 'roles.json', roles)
            print(json.dumps(record), flush=True)
    assert len(wrapper.aligned) == (len(batches) - 1) * method['settings']['latent_steps']
    actual, expected = torch.stack(wrapper.aligned), torch.stack(wrapper.reference_aligned)
    latent_record = difference(actual, expected)
    latent_record['allclose'] = torch.allclose(actual, expected, rtol=settings['rtol'], atol=settings['atol'])
    assert latent_record['allclose']
    leader_hidden = torch.stack(wrapper.hidden).clone()
    commands.broadcast_tensor(leader_hidden)
    assert torch.equal(leader_hidden, torch.stack(wrapper.hidden))
    decoder.initialize(transport)
    snapshot = decoder.capture_snapshot()
    decoder.step()
    eager = decoder.logits.clone()
    decoder.restore_capture_snapshot(snapshot)
    dist.barrier()
    decoder.capture(shared.runtime.graph_warmup_steps)
    decoder.graph.replay()
    assert torch.equal(eager, decoder.logits)
    decoder.restore_capture_snapshot(snapshot)
    events = [torch.cuda.Event(enable_timing=True) for _ in range(settings['decode_steps'] + 1)]
    tokens = []
    events[0].record()
    for _ in range(settings['decode_steps']):
        decoder.graph.replay()
        tokens.append(decoder.ids.clone())
        events[len(tokens)].record()
    torch.cuda.synchronize()
    record = {'rank': dist.get_rank(), 'world_size': dist.get_world_size(), 'task_ids': source['task_ids'],
        'batch_size': source['batch_size'], 'alignment': alignment_record, 'latent_vectors': latent_record,
        'latent_transitions': len(wrapper.aligned), 'hidden_equal_across_ranks': True,
        'roles': roles, 'eager_graph_exact': True, 'decode_steps': settings['decode_steps'],
        'decode_token_slots': settings['decode_steps'] * source['batch_size'],
        'median_itl_ms': statistics.median(left.elapsed_time(right) for left, right in zip(events, events[1:])),
        'output_ids': torch.cat(tokens, dim=-1).tolist(), 'task_accuracy_claimed': False}
    save(path / f'rank-{dist.get_rank()}.json', record)
    if commands.is_leader:
        save(path / 'summary.json', record)
    dist.barrier()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    settings = read(args.config)
    parallel = read(settings['parallel_config'])
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group(backend=parallel['tensor_backend'],
        timeout=timedelta(seconds=parallel['distributed_timeout_seconds']),
        device_id=torch.device('cuda', torch.cuda.current_device()))
    control = dist.new_group(backend=parallel['control_backend'],
        timeout=timedelta(seconds=parallel['distributed_timeout_seconds']))
    commands = ParallelCommands(control, dist.group.WORLD, parallel)
    run(settings, commands)
    dist.destroy_process_group()
