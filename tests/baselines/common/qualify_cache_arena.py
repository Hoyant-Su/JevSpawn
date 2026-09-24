import argparse
from collections import OrderedDict
import json
from pathlib import Path
import time

import torch
from transformers.cache_utils import LinearAttentionCacheLayerMixin

from baselines.common.config import SharedConfig
from baselines.common.tasks import read
from jev_spawn.infra.backend import Backend
from jev_spawn.runtime.cache_arena import StaticCacheArena
from jev_spawn.runtime.decoding import CapturedDecode


class FixedSteps:
    def __init__(self, width, steps):
        self.width, self.steps = width, steps

    def __call__(self, ids, scores):
        return torch.full((ids.shape[0],), ids.shape[1] - self.width >= self.steps,
                          device=ids.device, dtype=torch.bool)


def tensors(cache):
    result = []
    for layer in cache.layers:
        if isinstance(layer, LinearAttentionCacheLayerMixin):
            result.extend(layer.conv_states.values())
            result.extend(layer.recurrent_states.values())
        else:
            result.extend([layer.keys, layer.values, layer.cumulative_length])
    return result


def inputs(backend, rows, batch_size, width):
    # This is declared synthetic capacity stress built from real tokenizer outputs.
    encoded = backend.tokenizer([text for text in rows[:batch_size]], add_special_tokens=False)['input_ids']
    expanded = [(row * ((width + len(row) - 1) // len(row)))[:width] for row in encoded]
    ids = torch.tensor(expanded, device=backend.device, dtype=torch.long)
    return {'input_ids': ids, 'attention_mask': torch.ones_like(ids)}


def measure(decoder, model_inputs, options, steps, warmup_steps):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    output, events, capture = decoder.generate_tokens(model_inputs, options,
        FixedSteps(model_inputs['input_ids'].shape[1], steps), warmup_steps)
    torch.cuda.synchronize()
    return output, {'seconds': time.perf_counter() - started, 'steps': len(events),
                    'capture_seconds': capture, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                    'peak_reserved_bytes': torch.cuda.max_memory_reserved()}


@torch.inference_mode()
def qualify(specification, output):
    settings = read(specification)
    shared = SharedConfig.load(settings['shared_config'])
    output.mkdir(parents=True, exist_ok=False)
    (output / 'protocol.json').write_text(json.dumps(settings, indent=2) + '\n')
    backend = Backend(shared.backend())
    assert shared.generation.temperature == 0
    max_capacity = shared.model.max_input_tokens + shared.generation.max_new_tokens
    block = shared.runtime.graph_cache_block_tokens
    max_capacity = (max_capacity + block - 1) // block * block
    arena = StaticCacheArena(backend.model.config, shared.runtime.batch_size, max_capacity,
                             backend.model.lm_head.weight.dtype, backend.device)
    sources = read(settings['source_batches'])
    messages = sources[settings['source_batch_index']]['messages']
    assert len(messages) == shared.runtime.batch_size
    texts = backend.tokenizer.apply_chat_template(messages, tokenize=False,
                                                  add_generation_prompt=True, enable_thinking=False)
    options = {'max_new_tokens': shared.generation.max_new_tokens,
               'pad_token_id': backend.tokenizer.pad_token_id, 'do_sample': False}
    decoders, records = OrderedDict(), []

    def selected(batch_size, capacity):
        key = (batch_size, capacity)
        if key not in decoders:
            if len(decoders) == shared.runtime.graph_cache_size:
                decoders.popitem(last=False)
            decoders[key] = CapturedDecode(backend, batch_size, capacity, arena=arena)
        decoders.move_to_end(key)
        return decoders[key]

    for index, case in enumerate(settings['parity_cases']):
        batch_size, width = case['batch_size'], case['input_tokens']
        capacity = (width + options['max_new_tokens'] + block - 1) // block * block
        model_inputs = inputs(backend, texts, batch_size, width)
        independent = CapturedDecode(backend, batch_size, capacity)
        expected, baseline = measure(independent, model_inputs, options, case['decode_steps'],
                                     shared.runtime.graph_warmup_steps)
        current = selected(batch_size, capacity)
        actual, shared_measurement = measure(current, model_inputs, options, case['decode_steps'],
                                             shared.runtime.graph_warmup_steps)
        assert torch.equal(expected, actual), f'Token parity failed in case {index}.'
        for left, right in zip(tensors(independent.cache), tensors(current.cache), strict=True):
            assert left.shape == right.shape and left.stride() == right.stride() and left.dtype == right.dtype
            assert torch.equal(left, right), f'Cache state parity failed in case {index}.'
        assert torch.equal(independent.logits, current.logits)
        records.append({'case': case, 'exact_tokens_logits_cache_state': True, 'baseline': baseline,
                        'shared_arena': shared_measurement, 'resident_graphs': len(decoders)})
        (output / 'parity.json').write_text(json.dumps(records, indent=2) + '\n')
        del independent, expected, actual, current, model_inputs, left, right
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    case = settings['stress']
    assert case['batch_size'] == shared.runtime.batch_size
    assert case['input_tokens'] == shared.model.max_input_tokens
    assert case['decode_steps'] == shared.generation.max_new_tokens
    model_inputs = inputs(backend, texts, case['batch_size'], case['input_tokens'])
    decoder = selected(case['batch_size'], max_capacity)
    actual, stress = measure(decoder, model_inputs, options, case['decode_steps'], shared.runtime.graph_warmup_steps)
    assert stress['steps'] == case['decode_steps']
    assert actual.shape == (case['batch_size'], case['input_tokens'] + case['decode_steps'])
    result = {'runtime_version': 'shared_static_cache_arena_v1', 'qualification_only': True,
              'synthetic_stress': 'Real source token sequences repeated to exactly the declared full input width; no EOS early exit.',
              'arena_bytes': arena.nbytes, 'resident_graphs': len(decoders), 'parity': records,
              'stress': stress, 'output_shape': list(actual.shape), 'passed': True}
    (output / 'completion.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: result[key] for key in ['arena_bytes', 'resident_graphs', 'stress', 'passed']}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--specification', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    qualify(args.specification, args.output)


if __name__ == '__main__':
    main()
