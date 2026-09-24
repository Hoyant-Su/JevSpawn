import argparse
from dataclasses import asdict
from datetime import timedelta
import gc
import json
import os
from pathlib import Path
import statistics
import time

import torch
import torch.distributed as dist

from baselines.common.config import SharedConfig
from jev_spawn.infra.backend import Backend
from jev_spawn.infra.configuration import CORE, load_resource
from jev_spawn.infra.qwen35 import install_qwen35_execution
from jev_spawn.infra.qwen35.parallel import shard_qwen35
from jev_spawn.runtime.decoding import CapturedDecode


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


class ParallelDecode(CapturedDecode):
    def __init__(self, backend, batch_size, capacity):
        super().__init__(backend, batch_size, capacity)
        self.logits = torch.empty((batch_size, backend.model.lm_head.vocab_size),
                                 device=backend.device, dtype=backend.model.lm_head.weight.dtype)


@torch.inference_mode()
def run(settings):
    rank, size = dist.get_rank(), dist.get_world_size()
    assert size == settings['world_size']
    output = Path(settings['output'])
    if rank == settings['writer_rank']:
        output.mkdir(parents=True, exist_ok=False)
    dist.barrier()
    shared = SharedConfig.load(settings['shared_config'])
    CORE['backend']['device'] = str(torch.device('cuda', torch.cuda.current_device()))
    backend = Backend(shared.backend())
    execution = load_resource('qwen35_execution')
    original_bytes = sum(parameter.numel() * parameter.element_size() for parameter in backend.model.parameters())
    started = time.perf_counter()
    shard_qwen35(backend.model, execution, dist.group.WORLD)
    install_qwen35_execution(backend.model, execution)
    backend.config['execution'] = settings['optimized_execution']
    backend.config['world_size'] = size
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    partition_seconds = time.perf_counter() - started
    partition = {'rank': rank, 'world_size': size, 'gpu_name': torch.cuda.get_device_name(),
                 'original_parameter_bytes': original_bytes,
                 'local_parameter_bytes': sum(parameter.numel() * parameter.element_size()
                                             for parameter in backend.model.parameters()),
                 'partition_seconds': partition_seconds,
                 'linear_attention_heads': [{'layer': layer.linear_attn.layer_idx,
                    'key_heads': layer.linear_attn.num_k_heads, 'value_heads': layer.linear_attn.num_v_heads}
                    for layer in backend.model.model.language_model.layers if layer.block_type == 'linear_attention'],
                 'local_vocabulary_rows': backend.model.lm_head.weight.shape[0],
                 'global_vocabulary_rows': backend.model.lm_head.vocab_size}
    save(output / f'partition-rank{rank}.json', partition)
    source = json.loads(Path(settings['workload_config']).read_text())
    single_source = json.loads(Path(source['single_source']).read_text())
    ids = torch.tensor([single_source['input_ids'][source['single_row']]], device=backend.device)
    single = {'input_ids': ids, 'attention_mask': torch.ones_like(ids)}
    batch_source = json.loads(Path(source['batch_source']).read_text())[source['batch_index']]
    rendered = backend.tokenizer.apply_chat_template(batch_source['messages'], tokenize=False,
        add_generation_prompt=True, enable_thinking=False)
    batch = backend.tokenizer(rendered, padding=True, add_special_tokens=False,
                              return_tensors='pt', truncation=False).to(backend.device)
    assert batch['attention_mask'].sum(-1).tolist() == batch_source['input_tokens']
    workloads = [(single, [single_source['fields'][source['single_row']]['id']]),
                 (batch, batch_source['task_ids'])]
    records = []
    for inputs, identities in workloads:
        reference = json.loads((Path(settings['reference_run']) / f'native_b{len(identities)}.json').read_text())
        assert reference['task_ids'] == identities
        forced = torch.tensor([reference['initial_token_ids'], *zip(*reference['output_token_ids'])],
                              device=backend.device).unsqueeze(-1)
        capacity = inputs['input_ids'].shape[-1] + shared.generation.max_new_tokens
        decoder = ParallelDecode(backend, len(identities), capacity)
        torch.cuda.reset_peak_memory_stats()
        dist.barrier()
        started = time.perf_counter()
        decoder.prefill(inputs)
        torch.cuda.synchronize()
        prefill_seconds = time.perf_counter() - started
        initial_ids = decoder.ids.clone()
        snapshot = decoder.capture_snapshot()
        decoder.step()
        eager = decoder.logits.clone()
        decoder.restore_capture_snapshot(snapshot)
        dist.barrier()
        started = time.perf_counter()
        decoder.capture(shared.runtime.graph_warmup_steps)
        capture_seconds = time.perf_counter() - started
        decoder.graph.replay()
        exact = torch.equal(eager, decoder.logits)
        assert exact
        decoder.restore_capture_snapshot(snapshot)
        events = [torch.cuda.Event(enable_timing=True) for _ in range(settings['decode_steps'] + 1)]
        tokens = []
        dist.barrier()
        torch.cuda.synchronize()
        started = time.perf_counter()
        events[0].record()
        for step in range(settings['decode_steps']):
            decoder.ids.copy_(forced[step])
            decoder.graph.replay()
            events[step + 1].record()
            tokens.append(decoder.ids.clone())
        torch.cuda.synchronize()
        wall_seconds = time.perf_counter() - started
        local_intervals = torch.tensor([left.elapsed_time(right) for left, right in zip(events, events[1:])],
                                      device=backend.device, dtype=torch.float64)
        dist.all_reduce(local_intervals, op=dist.ReduceOp.MAX)
        metrics = torch.tensor([prefill_seconds, capture_seconds, wall_seconds,
                               torch.cuda.max_memory_allocated()], device=backend.device, dtype=torch.float64)
        dist.all_reduce(metrics, op=dist.ReduceOp.MAX)
        intervals = local_intervals.tolist()
        prefill_seconds, capture_seconds, wall_seconds, peak = metrics.tolist()
        actual_tokens = torch.cat(tokens, dim=-1)
        expected_tokens = torch.tensor(reference['output_token_ids'], device=backend.device)
        record = {'world_size': size, 'batch_size': len(identities), 'task_ids': identities,
            'input_tokens': inputs['attention_mask'].sum(-1).tolist(), 'cache_capacity': capacity,
            'prefill_seconds_max_rank': prefill_seconds, 'capture_seconds_max_rank': capture_seconds,
            'decode_seconds_max_rank': wall_seconds, 'per_step_ms_max_rank': intervals,
            'median_itl_ms': statistics.median(intervals),
            'per_row_tokens_per_second': settings['decode_steps'] / wall_seconds,
            'peak_allocated_bytes_max_rank': int(peak), 'eager_graph_exact': exact,
            'initial_tokens_match_native': (initial_ids.squeeze(-1) == forced[0].squeeze(-1)).tolist(),
            'output_token_ids': actual_tokens.tolist(), 'same_native_token_count': int((actual_tokens == expected_tokens).sum()),
            'token_slots': actual_tokens.numel(), 'collectives_included_in_timing': True}
        records.append(record)
        if rank == settings['writer_rank']:
            save(output / f'b{len(identities)}.json', record)
            print(json.dumps(record), flush=True)
        del decoder, snapshot, eager, tokens
        gc.collect()
        torch.cuda.empty_cache()
    if rank == settings['writer_rank']:
        save(output / 'summary.json', {'settings': settings, 'base_shared_configuration': asdict(shared),
            'records': records, 'scope': 'Matched full-model TP diagnostic; four H100 GPUs versus one in reference. No dataset accuracy claim.'})
    dist.barrier()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    settings = json.loads(args.config.read_text())
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group(backend=settings['distributed_backend'],
                            timeout=timedelta(seconds=settings['distributed_timeout_seconds']),
                            device_id=torch.device('cuda', int(os.environ['LOCAL_RANK'])))
    run(settings)
    dist.destroy_process_group()
