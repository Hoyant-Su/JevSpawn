import argparse
import json
from pathlib import Path
import statistics
import time

import torch
from transformers import GenerationConfig, StaticCache
from transformers.cache_utils import LinearAttentionCacheLayerMixin

from baselines.common.config import SharedConfig
from baselines.common.tasks import read, rows, render
from jev_spawn.infra.backend import Backend


class CapturedDecode:
    def __init__(self, backend, batch_size, capacity):
        self.backend = backend
        self.trunk = backend.model.model.language_model
        self.capacity = capacity
        self.cache = StaticCache(config=backend.model.config, max_cache_len=capacity)
        self.ids = torch.empty((batch_size, 1), device=backend.device, dtype=torch.long)
        self.positions = torch.empty_like(self.ids)
        self.key_valid = torch.ones((batch_size, capacity), device=backend.device, dtype=torch.bool)
        self.key_positions = torch.arange(capacity, device=backend.device)
        self.graph = None

    def prefill(self, inputs):
        self.cache.reset()
        width = inputs['input_ids'].shape[1]
        self.key_valid.fill_(True)
        self.key_valid[:, :width].copy_(inputs['attention_mask'])
        positions = inputs['attention_mask'].cumsum(1) - 1
        positions.masked_fill_(inputs['attention_mask'] == 0, 1)
        output = self.trunk(**inputs, position_ids=positions, past_key_values=self.cache, use_cache=True)
        logits = self.backend.model.lm_head(output.last_hidden_state[:, -1])
        self.ids.copy_(logits.argmax(-1)[:, None])
        self.positions.copy_(inputs['attention_mask'].sum(1)[:, None])
        return self.ids[:, 0].clone()

    def step(self):
        valid = self.key_valid & (self.key_positions <= self.cache.get_seq_length())
        output = self.trunk(input_ids=self.ids, position_ids=self.positions,
                            attention_mask={'full_attention': valid[:, None, None, :], 'linear_attention': None},
                            past_key_values=self.cache, use_cache=True)
        self.logits = self.backend.model.lm_head(output.last_hidden_state[:, -1])
        self.ids.copy_(self.logits.argmax(-1)[:, None])
        self.positions.add_(1)

    def capture(self, warmup_steps):
        state = [self.ids, self.positions]
        for layer in self.cache.layers:
            if isinstance(layer, LinearAttentionCacheLayerMixin):
                state.extend(layer.conv_states.values())
                state.extend(layer.recurrent_states.values())
            else:
                state.extend([layer.keys, layer.values, layer.cumulative_length])
        saved = [tensor.clone() for tensor in state]

        def reset():
            for destination, source in zip(state, saved):
                destination.copy_(source)

        stream = torch.cuda.Stream(device=self.backend.device)
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(warmup_steps):
                reset()
                self.step()
            reset()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=stream):
            self.step()
        reset()
        torch.cuda.synchronize()

    def generate(self, inputs, tokens, replay):
        torch.cuda.synchronize()
        started = time.perf_counter()
        token = self.prefill(inputs)
        events = []
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        events.append(event)
        outputs = [token]
        eos = torch.tensor(self.backend.eos_ids, device=token.device)
        finished = torch.isin(token, eos)
        for _ in range(tokens - 1):
            if bool(finished.all()):
                break
            if replay:
                self.graph.replay()
            else:
                self.step()
            token = self.ids[:, 0].clone()
            token.masked_fill_(finished, self.backend.tokenizer.pad_token_id)
            outputs.append(token)
            finished |= torch.isin(token, eos)
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            events.append(event)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        sequences = torch.stack(outputs, 1).tolist()
        sequences = [sequence[:next((i + 1 for i, token in enumerate(sequence)
                                     if token in self.backend.eos_ids), len(sequence))] for sequence in sequences]
        intervals = [a.elapsed_time(b) for a, b in zip(events, events[1:])]
        return {'elapsed_seconds': elapsed, 'output_ids': sequences,
                'itl_median_ms': statistics.median(intervals) if intervals else None,
                'itl_max_ms': max(intervals) if intervals else None,
                'graph_replays': len(outputs) - 1 if replay else 0}


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--shared-config', type=Path, required=True)
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config, profile = SharedConfig.load(args.shared_config), read(args.profile)
    args.output.mkdir(parents=True, exist_ok=False)
    backend = Backend(config.backend())
    tasks = rows(profile['tasks'])[profile['offset']:profile['offset'] + profile['samples']]
    texts = backend.tokenizer.apply_chat_template(
        [[{'role': 'user', 'content': render(task)}] for task in tasks],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    records = []
    for size in profile['batch_sizes']:
        inputs, lengths = backend._encode(texts[:size])
        capacity = inputs['input_ids'].shape[1] + profile['tokens']
        decoder = CapturedDecode(backend, size, capacity)
        reference = backend.model.generate(**inputs, logits_to_keep=1,
            generation_config=GenerationConfig(do_sample=False, max_new_tokens=profile['tokens'],
                use_cache=True, disable_compile=True, cache_implementation='static',
                eos_token_id=backend.eos_ids, pad_token_id=backend.tokenizer.pad_token_id,
                bos_token_id=backend.tokenizer.bos_token_id))[:, inputs['input_ids'].shape[1]:].tolist()
        reference = [row[:next((i + 1 for i, token in enumerate(row) if token in backend.eos_ids), len(row))]
                     for row in reference]
        eager = decoder.generate(inputs, profile['tokens'], False)
        (args.output / f'eager-{size}.json').write_text(json.dumps({'reference': reference, **eager}, indent=2) + '\n')
        assert eager['output_ids'] == reference, 'Explicit text masks or cache path changed reference tokens.'
        decoder.prefill(inputs)
        started = time.perf_counter()
        decoder.capture(profile['capture_warmup_steps'])
        capture_seconds = time.perf_counter() - started
        for repeat in range(profile['repeats']):
            eager = decoder.generate(inputs, profile['tokens'], False)
            replay = decoder.generate(inputs, profile['tokens'], True)
            record = {'batch_size': size, 'repeat': repeat, 'capture_seconds': capture_seconds,
                      'eager': eager, 'replay': replay,
                      'exact_token_parity': eager['output_ids'] == replay['output_ids'] == reference}
            records.append(record)
            (args.output / 'measurements.json').write_text(json.dumps(records, indent=2) + '\n')
            print(json.dumps({key: value for key, value in record.items() if key not in ('eager', 'replay')} |
                             {'eager_seconds': eager['elapsed_seconds'], 'replay_seconds': replay['elapsed_seconds'],
                              'replay_itl_ms': replay['itl_median_ms']}), flush=True)
            assert record['exact_token_parity'], 'Graph replay changed reference tokens.'
    (args.output / 'result.json').write_text(json.dumps({'config': profile, 'measurements': records,
        'scope': 'Static decode infrastructure qualification. Capture is separately reported and must be amortized by reuse.'}, indent=2) + '\n')


if __name__ == '__main__':
    main()
