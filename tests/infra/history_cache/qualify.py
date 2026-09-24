import argparse
import gc
import json
from pathlib import Path
import time

import torch
import torch.distributed as dist
from transformers.cache_utils import StaticLayer

from baselines.common.config import SharedConfig
from baselines.common.parallel_run import initialize_parallel
from jev_spawn.algo.structured import padded
from jev_spawn.infra.history_prefill import HistoryDecode, HistoryPrefill
from jev_spawn.infra.kernel_tuning import load_tuning
from jev_spawn.runtime.cache_arena import StaticCacheArena
from jev_spawn.runtime.decoding import CapturedDecode


def snapshot(decoder, lengths):
    width = int(decoder.cache.get_seq_length())
    result = []
    for layer in decoder.cache.layers:
        if type(layer) is StaticLayer:
            result.extend(('kv', getattr(layer, name)[row:row + 1, :, width - length:width].clone())
                          for name in ('keys', 'values') for row, length in enumerate(lengths))
        else:
            result.extend((name, tensor.clone()) for name in ('conv_states', 'recurrent_states')
                          for tensor in getattr(layer, name).values())
    return result


def differences(left, right):
    values = {}
    for (name, first), (other, second) in zip(left, right, strict=True):
        assert name == other and first.shape == second.shape and first.dtype == second.dtype
        values.setdefault(name, []).append((first.float() - second.float()).abs().max())
    return {name: float(torch.stack(items).max()) for name, items in values.items()}


class ObservedDecode(CapturedDecode):
    def prefill(self, inputs):
        torch.cuda.synchronize(self.backend.device)
        start = time.perf_counter()
        value = super().prefill(inputs)
        torch.cuda.synchronize(self.backend.device)
        self.prefill_seconds = time.perf_counter() - start
        self.first_logits = self.logits.clone()
        self.prefill_states = snapshot(self, inputs['attention_mask'].sum(-1).tolist())
        return value


class ObservedHistoryDecode(HistoryDecode):
    def prefill(self, inputs):
        torch.cuda.synchronize(self.backend.device)
        start = time.perf_counter()
        value = super().prefill(inputs)
        torch.cuda.synchronize(self.backend.device)
        self.prefill_seconds = time.perf_counter() - start
        self.first_logits = self.logits.clone()
        self.prefill_states = snapshot(self, inputs['attention_mask'].sum(-1).tolist())
        return value


def run(config):
    shared = SharedConfig.load(config['shared_config'])
    backend, commands, startup = initialize_parallel(shared, json.loads(Path(config['parallel_settings']).read_text()))
    tuning = load_tuning(Path(config['cache_directory']) / config['cache_file'].format(rank=dist.get_rank()),
                         backend, json.loads(Path(config['tuning_settings']).read_text()))
    history = HistoryPrefill(backend, json.loads(Path(config['history_settings']).read_text()))
    output = Path(config['output'])
    output.mkdir(parents=True, exist_ok=True)
    source = json.loads(Path(config['source']).read_text())
    decoders, arenas, previous = {}, [], {}

    @torch.inference_mode()
    def compute(workload):
        batch = source[workload['batch']]
        messages = [batch['messages'][row] for row in workload['rows']]
        text = backend.tokenizer.apply_chat_template(messages, tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
        sequences = backend.tokenizer(text, add_special_tokens=False, truncation=False)['input_ids']
        expected = [batch['input_tokens'][row] for row in workload['rows']]
        assert list(map(len, sequences)) == expected and max(expected) <= shared.model.max_input_tokens
        ids, mask = padded(sequences, backend.tokenizer.pad_token_id, backend.device, 'left')
        capacity = shared.model.max_input_tokens + config['generation_tokens']
        size = len(sequences)
        reports, states, logits, decoded = {}, {}, {}, {}
        for name, constructor in [('native', ObservedDecode), ('history', ObservedHistoryDecode)]:
            key = (name, size)
            if key not in decoders:
                arena = StaticCacheArena(backend.cache_config, size, capacity,
                    backend.model.lm_head.weight.dtype, backend.device)
                arenas.append(arena)
                decoders[key] = constructor(backend, size, capacity, arena=arena)
            decoder = decoders[key]
            if name == 'history':
                decoder.history = history
                history.sequences = sequences
            budgets = torch.tensor(config['row_budgets'][:size], device=backend.device)
            eos = torch.tensor(backend.eos_ids, device=backend.device)
            natural_ends = set()
            deliveries, delivered_states = {}, {}

            def stop(tokens, scores):
                natural = torch.isin(tokens[:, -1], eos)
                natural_ends.update(index for index, done in enumerate(natural.tolist()) if done)
                return natural | (tokens.shape[1] - ids.shape[1] >= budgets)

            def deliver(tokens, indices):
                for index in indices:
                    deliveries[index] = tokens[index].clone()
                if name == 'history':
                    for index in indices:
                        key = tuple([*sequences[index], *tokens[index].tolist()[:-1]])
                        cache = history.cache.read(key)
                        assert cache.get_seq_length() == len(key)
                        delivered_states[index] = len(key)

            options = {'max_new_tokens': config['generation_tokens'], 'pad_token_id': backend.tokenizer.pad_token_id,
                       'do_sample': config['do_sample'], 'temperature': shared.generation.temperature}
            torch.cuda.reset_peak_memory_stats(backend.device)
            dist.barrier()
            start = time.perf_counter()
            generated, events, capture = decoder.generate_tokens({'input_ids': ids, 'attention_mask': mask},
                options, stop, shared.runtime.graph_warmup_steps, on_tokens=deliver)
            torch.cuda.synchronize(backend.device)
            elapsed = time.perf_counter() - start
            intervals = [first.elapsed_time(second) for first, second in zip(events, events[1:])]
            decoded[name] = [deliveries[index].tolist() for index in range(size)]
            states[name], logits[name] = decoder.prefill_states, decoder.first_logits
            reports[name] = {'seconds_including_diagnostic_snapshots': elapsed,
                'prefill_seconds_excluding_diagnostic_snapshots': decoder.prefill_seconds,
                'graph_capture_seconds': capture, 'decode_intervals_ms': intervals,
                'output_token_ids': decoded[name], 'natural_eos_rows': sorted(natural_ends),
                'saved_generated_prefix_lengths': delivered_states,
                'peak_allocated_bytes': torch.cuda.max_memory_allocated(backend.device),
                'history_work': history.records[-1] if name == 'history' else None}
            del decoder.prefill_states, decoder.first_logits
        difference = float((logits['native'].float() - logits['history'].float()).abs().max())
        differences_by_state = differences(states['native'], states['history'])
        rank_records = [None for _ in range(shared.runtime.world_size)]
        dist.all_gather_object(rank_records, decoded, group=commands.control_group)
        assert all(item == decoded for item in rank_records)
        ranks = torch.tensor([reports[name]['seconds_including_diagnostic_snapshots'] for name in reports],
                              device=backend.device, dtype=torch.float64)
        dist.all_reduce(ranks, op=dist.ReduceOp.MAX)
        result = {'name': workload['name'], 'task_ids': [batch['task_ids'][row] for row in workload['rows']],
            'source_batch': workload['batch'], 'source_rows': workload['rows'], 'input_tokens': expected,
            'reports': reports, 'rankmax_seconds': dict(zip(reports, ranks.tolist(), strict=True)),
            'allrank_output_tokens_identical': True, 'native_history_tokens_equal': decoded['native'] == decoded['history'],
            'prefill_full_head_max_abs_difference': difference, 'prefill_state_max_abs_difference': differences_by_state}
        signature = tuple(map(tuple, sequences))
        if signature in previous:
            original = previous[signature]
            result['history_repeat_logits_exact'] = torch.equal(original, logits['history'])
        previous[signature] = logits['history'].clone()
        (output / config['rank_file'].format(name=workload['name'], rank=dist.get_rank())).write_text(json.dumps(result, indent=2) + '\n')
        return result

    commands.register(config['command'], compute)
    if commands.is_leader:
        try:
            reports = [commands.call(config['command'], workload) for workload in config['workloads']]
            Path(config['result']).write_text(json.dumps({'config': config, 'startup': startup,
                'tuning': tuning, 'reports': reports, 'scope': 'Fixed real prompt operator diagnostic. Generation caps and snapshots are explicit; not task quality or production timing.'}, indent=2) + '\n')
        finally:
            commands.finish()
    else:
        commands.serve()
    decoders.clear()
    arenas.clear()
    previous.clear()
    history.cache.clear()
    gc.collect()
    dist.destroy_process_group(commands.control_group)
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    run(json.loads(Path(parser.parse_args().config).read_text()))
