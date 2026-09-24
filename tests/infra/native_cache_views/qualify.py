import argparse
from functools import partial
import json
from pathlib import Path
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers.cache_utils import DynamicLayer

from baselines.common.config import SharedConfig
from baselines.common.parallel_run import initialize_parallel
from candidate import split_compact
from jev_spawn.algo.structured import padded
from jev_spawn.runtime.native_cache_batch import pack_native_caches, split_native_cache_at


def tensors(states):
    return [tensor for state in states for layer in state.layers for tensor in
            ([layer.keys, layer.values] if type(layer) is DynamicLayer else
             [*layer.conv_states.values(), *layer.recurrent_states.values()])]


def storage_bytes(states):
    storages = {tensor.untyped_storage().data_ptr(): tensor.untyped_storage().nbytes()
                for tensor in tensors(states)}
    return sum(storages.values())


def identical(left, right):
    pairs = list(zip(tensors(left), tensors(right), strict=True))
    return all(torch.equal(a, b) for a, b in pairs)


@torch.inference_mode()
def run(settings):
    shared = SharedConfig.load(settings['shared_config'])
    backend, commands, startup = initialize_parallel(
        shared, json.loads(Path(settings['parallel_settings']).read_text()))
    rows = []
    compact = partial(split_compact, settings=json.loads(Path(settings['state_copy_settings']).read_text()))
    for source in settings['sources']:
        requests = json.loads(Path(source).read_text())['requests']
        sequences = [request['input_ids'] for request in requests]
        prefixes = [sequence[:-1] for sequence in sequences]
        lengths = list(map(len, prefixes))
        ids, mask = padded(prefixes, backend.tokenizer.pad_token_id, backend.device, 'left')
        output = backend.model.model(input_ids=ids, attention_mask=mask,
            position_ids=(mask.cumsum(-1) - 1).clamp_min(0), use_cache=True)
        original = output.past_key_values
        stops = [original.get_seq_length()] * len(lengths)
        cloned = split_native_cache_at(original, lengths, stops)
        compacted = compact(original, lengths, stops)
        assert identical(cloned, compacted)
        measurements, logits, packed = {}, {}, {}
        for name, split in [('clone', split_native_cache_at), ('compact', compact)]:
            for _ in range(settings['warmup_steps']):
                transient = split(original, lengths, stops)
                del transient
            timings = []
            for _ in range(settings['repetitions']):
                torch.cuda.synchronize(backend.device)
                started = time.perf_counter()
                states = split(original, lengths, stops)
                torch.cuda.synchronize(backend.device)
                split_seconds = time.perf_counter() - started
                started = time.perf_counter()
                cache, prefix_mask = pack_native_caches(states)
                torch.cuda.synchronize(backend.device)
                timings.append({'split_seconds': split_seconds,
                                'pack_seconds': time.perf_counter() - started})
                del states, cache, prefix_mask
            measurements[name] = timings
            states = {'clone': cloned, 'compact': compacted}[name]
            cache, prefix_mask = pack_native_caches(states)
            source_pointers = {t.untyped_storage().data_ptr() for t in tensors(states)}
            assert not source_pointers.intersection(t.untyped_storage().data_ptr() for t in tensors([cache]))
            packed[name] = cache
        assert identical([packed['clone']], [packed['compact']])
        for name, cache in packed.items():
            last = torch.tensor([sequence[-1] for sequence in sequences], device=backend.device)[:, None]
            full_mask = torch.cat((prefix_mask, torch.ones_like(last)), dim=-1)
            positions = torch.tensor(lengths, device=backend.device)[:, None]
            result = backend.model.model(input_ids=last, attention_mask=full_mask,
                position_ids=positions, past_key_values=cache, use_cache=True)
            logits[name] = F.linear(result.last_hidden_state[:, -1].float(), backend.finite_output_weights)
        tensors_equal_after_forward = identical(cloned, compacted)
        original_unchanged = identical(cloned, split_native_cache_at(original, lengths, stops))
        logits_equal = torch.equal(logits['clone'], logits['compact'])
        row = {'source': source, 'task_ids': [r['task_id'] for r in requests],
            'batch_size': len(requests), 'prefix_lengths': lengths,
            'measurements': measurements, 'tensor_equal_before_forward': True,
            'tensor_equal_after_forward': tensors_equal_after_forward,
            'original_prefill_cache_unchanged': original_unchanged,
            'packed_inputs_equal': True, 'packed_storage_disjoint_from_sources': True,
            'logits_equal': logits_equal,
            'max_logit_error': (logits['clone'] - logits['compact']).abs().max().item(),
            'retained_storage_bytes': {'clone_all_rows': storage_bytes(cloned),
                'compact_all_rows': storage_bytes(compacted),
                'clone_one_row': storage_bytes(cloned[:1]), 'compact_one_row': storage_bytes(compacted[:1])}}
        rows.append(row)
        destination = Path(settings['output'])
        destination.mkdir(parents=True, exist_ok=True)
        (destination / f'rank-{dist.get_rank()}.json').write_text(json.dumps({
            'settings': settings, 'startup': startup, 'rows': rows,
            'scope': 'Real recorded prompt tokens and real model-produced native states. Row-owned buffers preserve compact ownership and copy exact tensors using the existing grouped GPU copy operator. Not end-to-end rollout timing.'}, indent=2) + '\n')
        assert tensors_equal_after_forward and original_unchanged and logits_equal
        del original, cloned, compacted, packed, output, result, cache, logits
    dist.destroy_process_group(commands.control_group)
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    run(json.loads(parser.parse_args().config.read_text()))
