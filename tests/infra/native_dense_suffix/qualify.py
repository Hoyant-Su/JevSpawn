import argparse
import json
from pathlib import Path
from statistics import median

import torch
import torch.distributed as dist

from baselines.common.config import SharedConfig
from baselines.common.parallel_run import initialize_parallel
from candidate import dense_suffix_forward
from jev_spawn.algo.structured import padded
from jev_spawn.infra.cached_attention import RaggedCacheAttention
from jev_spawn.infra.qwen35 import gdn
from jev_spawn.runtime.batched_state_copy import copy_states
from jev_spawn.runtime.native_cache_batch import pack_native_caches, split_native_cache
from jev_spawn.runtime.ragged_suffix import RaggedSuffix
from tests.infra.native_suffix_graph.qualify import compare, measure, next_logits, tensors, wrapper


@torch.inference_mode()
def run(settings):
    shared = SharedConfig.load(settings['shared_config'])
    backend, commands, startup = initialize_parallel(
        shared, json.loads(Path(settings['parallel_settings']).read_text()))
    destination = Path(settings['output'])
    destination.mkdir(parents=True, exist_ok=True)
    copy_settings = json.loads(Path(settings['state_copy_settings']).read_text())
    implementations = {'ragged': gdn.ragged_gdn_forward, 'dense': dense_suffix_forward}
    rows = []
    for source in settings['sources']:
        requests = json.loads(Path(source).read_text())['requests']
        roots = [request['root_tokens'] for request in requests]
        sequences = [request['input_ids'] for request in requests]
        tails = [sequence[len(root):-1] for sequence, root in zip(sequences, roots, strict=True)]
        assert all(sequence[:len(root)] == root for sequence, root in zip(sequences, roots, strict=True))
        root_ids, root_mask = padded(roots, backend.tokenizer.pad_token_id, backend.device, 'left')
        root = backend.model.model(input_ids=root_ids, attention_mask=root_mask,
            position_ids=(root_mask.cumsum(-1) - 1).clamp_min(0), use_cache=True)
        states = split_native_cache(root.past_key_values, list(map(len, roots)))
        immutable, prefix_mask = pack_native_caches(states)
        working, _ = pack_native_caches(states)
        snapshots = [tensor.clone() for tensor in tensors(immutable)]
        pairs = list(zip(tensors(working), tensors(immutable), strict=True))
        ids, mask = padded(tails, backend.tokenizer.pad_token_id, backend.device, 'right')
        full_mask = torch.cat((prefix_mask, mask), dim=-1)
        positions = (full_mask.cumsum(-1) - 1).clamp_min(0)[:, -ids.shape[1]:]
        lengths = list(map(len, tails))
        descriptor = RaggedSuffix(lengths, backend.device)
        attention = RaggedCacheAttention(descriptor, list(map(len, roots)))

        def forward():
            copy_states(pairs, copy_settings)
            return backend.model.model(input_ids=ids, attention_mask=full_mask,
                position_ids=positions, past_key_values=wrapper(working), use_cache=True,
                ragged_suffix=descriptor, decode_attention=attention)

        outputs, timings = {}, {}
        for name, implementation in implementations.items():
            gdn.ragged_gdn_forward = implementation
            output = forward()
            outputs[name] = {
                'hidden': descriptor.pack(output.last_hidden_state).clone(),
                'cache': [tensor.clone() for tensor in tensors(output.past_key_values)],
                'logits': next_logits(backend, output.past_key_values, list(map(len, roots)),
                                     lengths, prefix_mask.shape[1], sequences)}
            for _ in range(shared.runtime.graph_warmup_steps):
                forward()
            timings[name] = measure(forward, backend.device, settings['repetitions'])
        checks = {key: compare([outputs['ragged'][key]], [outputs['dense'][key]])
                  for key in ('hidden', 'logits')}
        checks['cache'] = compare(outputs['ragged']['cache'], outputs['dense']['cache'])
        checks['source'] = compare(snapshots, tensors(immutable))
        counts = torch.tensor([len(request['field']['options']) for request in requests], device=backend.device)
        valid = torch.arange(outputs['ragged']['logits'].shape[-1], device=backend.device)[None] < counts[:, None]
        choices = {name: value['logits'].masked_fill(~valid, -torch.inf).argmax(-1).tolist()
                   for name, value in outputs.items()}
        row = {'source': source, 'batch_size': len(requests), 'suffix_lengths': lengths,
               'checks': checks, 'choices': choices, 'timings': timings,
               'median_seconds': {name: median(sample['wall_seconds'] for sample in samples)
                                  for name, samples in timings.items()}}
        rows.append(row)
        (destination / f'rank-{dist.get_rank()}.json').write_text(json.dumps(
            {'settings': settings, 'startup': startup, 'rows': rows}, indent=2) + '\n')
    gdn.ragged_gdn_forward = implementations['ragged']
    dist.destroy_process_group(commands.control_group)
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    run(json.loads(parser.parse_args().config.read_text()))
