from copy import copy
import time

import torch
from transformers.cache_utils import DynamicCache, DynamicLayer, LinearAttentionLayer, StaticLayer

from jev_spawn.algo.structured import padded
from jev_spawn.infra.cached_suffix import ragged_suffix
from jev_spawn.runtime.batched_state_copy import copy_states
from jev_spawn.runtime.decoding import CapturedDecode
from jev_spawn.runtime.history_cache import HistoryCache
from jev_spawn.runtime.native_cache_batch import _metadata, _with_layers, pack_native_caches, split_native_cache
from jev_spawn.runtime.native_restore import load_native_prefixes


class HistoryPrefill:
    extend_states = staticmethod(ragged_suffix)

    def __init__(self, backend, settings):
        self.backend, self.settings = backend, settings
        self.cache = HistoryCache(settings['cache'])
        self.records = []

    def cold_prefill(self, decoder, inputs):
        return CapturedDecode.prefill(decoder, inputs)

    @torch.inference_mode()
    def compute(self, sequences):
        started = time.perf_counter()
        unique = list(dict.fromkeys(map(tuple, sequences)))
        prefixes = [self.cache.prefix(sequence) for sequence in unique]
        plan = {'prefixes': prefixes, 'entries': len(self.cache.entries), 'bytes': self.cache.bytes}
        commands = self.backend.parallel_commands
        synchronized = commands.leader_value(plan if commands.is_leader else None)
        assert plan == synchronized, 'Tensor-parallel history cache state diverged.'
        work = {'computed_input_tokens': self.settings['zero'], 'padded_input_tokens': self.settings['zero']}
        states = {sequence: self.cache.read(prefix) for sequence, prefix in zip(unique, prefixes, strict=True) if prefix}
        cold = [sequence for sequence, prefix in zip(unique, prefixes, strict=True) if not prefix]
        if cold:
            ids, mask = padded([list(sequence) for sequence in cold], self.backend.tokenizer.pad_token_id, self.backend.device, 'left')
            output = self.backend.model.model(input_ids=ids, attention_mask=mask,
                position_ids=(mask.cumsum(-1) - self.settings['position_step']).clamp_min(self.settings['zero']),
                use_cache=True)
            states.update(zip(cold, split_native_cache(output.past_key_values, list(map(len, cold))), strict=True))
            work['computed_input_tokens'] += sum(map(len, cold))
            work['padded_input_tokens'] += ids.numel()
        warm = [(sequence, prefix) for sequence, prefix in zip(unique, prefixes, strict=True) if prefix]
        if warm:
            extended = self.extend_states(self.backend, [states[sequence] for sequence, _ in warm],
                                          [list(sequence[len(prefix):]) for sequence, prefix in warm], work)
            states.update(zip([sequence for sequence, _ in warm], extended, strict=True))
        for sequence in unique:
            self.cache.store(sequence, states[sequence], None)
        torch.cuda.synchronize(self.backend.device)
        self.records.append({'logical_input_tokens': sum(map(len, sequences)),
            'unique_input_tokens': sum(map(len, unique)), 'matched_prefix_tokens': sum(map(len, prefixes)),
            'requests': len(sequences), 'unique_requests': len(unique), 'cold_rows': len(cold),
            'warm_rows': len(warm), 'cache_bytes': self.cache.bytes,
            'elapsed_seconds': time.perf_counter() - started, **work})
        return [states[tuple(sequence)] for sequence in sequences]

    @torch.inference_mode()
    def save_generated(self, decoder, tokens, indices):
        tails = tokens[indices].tolist()
        sequences = [[*decoder.sequences[row], *tail[:self.settings['exclude_last']]]
                     for row, tail in zip(indices, tails, strict=True)]
        self.save_rows(decoder, sequences, indices)

    @torch.inference_mode()
    def save_rows(self, decoder, sequences, indices):
        width = decoder.cache.get_seq_length().item()
        pairs, saved = [], []
        for row, sequence in zip(indices, sequences, strict=True):
            length = len(sequence)
            template = DynamicCache(config=self.backend.cache_config)
            layers = []
            for original, current in zip(template.layers, decoder.cache.layers, strict=True):
                layer = copy(original)
                tensor_names = ('keys', 'values') if type(original) is DynamicLayer else ('conv_states', 'recurrent_states')
                layer.__dict__ = _metadata(original if type(original) is DynamicLayer else current, tensor_names)
                if type(original) is DynamicLayer:
                    layer.dtype, layer.device = current.keys.dtype, current.keys.device
                    layer.is_initialized = current.is_initialized
                    assert type(current) is StaticLayer
                    for name in tensor_names:
                        source = getattr(current, name)[row:row + self.settings['row_step'], :, width - length:width]
                        destination = torch.empty_like(source)
                        setattr(layer, name, destination)
                        pairs.append((destination, source))
                else:
                    assert type(original) is type(current) is LinearAttentionLayer
                    for name in tensor_names:
                        target = {}
                        for key, tensor in getattr(current, name).items():
                            source = tensor[row:row + self.settings['row_step']]
                            target[key] = torch.empty_like(source)
                            pairs.append((target[key], source))
                        setattr(layer, name, target)
                layers.append(layer)
            logits = torch.empty_like(decoder.logits[row:row + self.settings['row_step']])
            pairs.append((logits, decoder.logits[row:row + self.settings['row_step']]))
            saved.append((sequence, _with_layers(template, layers), logits))
        copy_states(pairs, self.settings['state_copy'])
        for sequence, state, logits in saved:
            self.cache.store(sequence, state, logits)


class HistoryDecode(CapturedDecode):
    @torch.inference_mode()
    def prefill(self, inputs):
        policy = self.history.settings
        self.sequences = self.history.sequences
        prefixes = [self.history.cache.prefix(sequence[:policy['exclude_last']]) for sequence in self.sequences]
        complete = [tuple(sequence) in self.history.cache.outputs
                    and self.history.cache.outputs[tuple(sequence)] is not None for sequence in self.sequences]
        commands = self.backend.parallel_commands
        plan = (prefixes, complete)
        synchronized = commands.leader_value(plan if commands.is_leader else None)
        assert plan == synchronized, 'Tensor-parallel history cache lookup diverged.'
        if not any(prefixes) and not any(complete):
            started = time.perf_counter()
            token = self.history.cold_prefill(self, inputs)
            torch.cuda.synchronize(self.backend.device)
            self.history.records.append({'logical_input_tokens': sum(map(len, self.sequences)),
                'unique_input_tokens': sum(map(len, dict.fromkeys(map(tuple, self.sequences)))),
                'matched_prefix_tokens': policy['zero'], 'requests': len(self.sequences),
                'unique_requests': len(set(map(tuple, self.sequences))), 'cold_rows': len(self.sequences),
                'warm_rows': policy['zero'], 'cache_bytes': self.history.cache.bytes,
                'elapsed_seconds': time.perf_counter() - started,
                'computed_input_tokens': sum(map(len, self.sequences)),
                'padded_input_tokens': inputs['input_ids'].numel(), 'execution': 'native_full_prefill'})
            self.history.save_rows(self, self.sequences, list(range(len(self.sequences))))
            self.history.records[-policy['record_offset']].update(cache_bytes=self.history.cache.bytes,
                elapsed_seconds=time.perf_counter() - started)
            return token
        started = time.perf_counter()
        states = {row: self.history.cache.read(tuple(sequence))
                  for row, (sequence, hit) in enumerate(zip(self.sequences, complete, strict=True)) if hit}
        logits = {row: self.history.cache.outputs[tuple(self.sequences[row])] for row in states}
        pending = [row for row, hit in enumerate(complete) if not hit]
        work = {'computed_input_tokens': policy['zero'], 'padded_input_tokens': policy['zero'],
                'matched_prefix_tokens': policy['zero'], 'cold_rows': policy['zero'], 'warm_rows': policy['zero']}
        if pending:
            suffix_states = self.history.compute([self.sequences[row][:policy['exclude_last']] for row in pending])
            work.update(self.history.records.pop())
            cache, mask = pack_native_caches(suffix_states)
            ids = torch.tensor([self.sequences[row][policy['last_index']] for row in pending],
                               device=self.backend.device)[:, None]
            positions = mask.sum(-1)[:, None]
            output = self.backend.model.model(input_ids=ids, position_ids=positions,
                attention_mask=torch.cat((mask, torch.ones_like(ids)), dim=-1),
                past_key_values=cache, use_cache=True)
            extended = split_native_cache(output.past_key_values, [len(self.sequences[row]) for row in pending])
            states.update(zip(pending, extended, strict=True))
            scores = self.backend.model.lm_head(output.last_hidden_state[:, policy['last_index']])
            logits.update((row, scores[index:index + policy['row_step']]) for index, row in enumerate(pending))
            work['computed_input_tokens'] += len(pending)
            work['padded_input_tokens'] += len(pending)
            for row in pending:
                self.history.cache.store(self.sequences[row], states[row], logits[row].clone())
        self.logits.copy_(torch.cat([logits[row] for row in range(len(self.sequences))]))
        chosen = self.logits.argmax(-1)
        load_native_prefixes(self, [states[row] for row in range(len(self.sequences))],
                             chosen.tolist(), policy['state_copy'])
        self.ids.copy_(self.logits.argmax(-1)[:, None])
        torch.cuda.synchronize(self.backend.device)
        exact_tokens = sum(len(sequence) for sequence, hit in zip(self.sequences, complete, strict=True) if hit)
        self.history.records.append({**work, 'logical_input_tokens': sum(map(len, self.sequences)),
            'matched_prefix_tokens': work['matched_prefix_tokens'] + exact_tokens,
            'requests': len(self.sequences), 'exact_rows': sum(complete),
            'cache_bytes': self.history.cache.bytes, 'elapsed_seconds': time.perf_counter() - started,
            'execution': 'native_cached_prefill'})
        return self.ids[:, policy['first_index']].clone()

    def generate_tokens(self, inputs, options, stopping, warmup_steps, on_tokens=None, between_steps=None):
        assert on_tokens is not None, 'History checkpoints require immediate row-completion delivery.'

        def deliver(tokens, indices):
            self.history.save_generated(self, tokens, indices)
            on_tokens(tokens, indices)

        return super().generate_tokens(inputs, options, stopping, warmup_steps, deliver, between_steps)
