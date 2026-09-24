import time

import torch
from transformers.cache_utils import DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast

from baselines.latentmas.native_hybrid_transport import CapturedPaddingTransport
from jev_spawn.algo.structured import padded
from jev_spawn.infra.history_prefill import HistoryPrefill
from jev_spawn.runtime.native_cache_batch import pack_native_caches, split_native_cache
from jev_spawn.runtime.native_restore import load_native_prefixes


class RoleZeroHistory:
    def __init__(self, backend, settings):
        self.backend, self.settings = backend, settings
        self.history = HistoryPrefill(backend, settings['history'])
        self.records = []

    @torch.inference_mode()
    def prefill(self, transport, sequences):
        assert transport.cache is None and transport.mask is None
        backend, policy, cache = self.backend, self.settings, self.history.cache
        sequences = list(map(tuple, sequences))
        keys = sequences
        prefixes = [cache.prefix(key) for key in keys]
        commands = backend.parallel_commands
        plan = (keys, prefixes)
        assert plan == commands.leader_value(plan if commands.is_leader else None)
        complete = [key for key in keys if key in cache.outputs and cache.outputs[key] is not None]
        cold = [key for key, prefix in zip(keys, prefixes, strict=True) if not prefix]
        partial = [key for key in keys if key not in complete and key not in cold]
        states = {key: cache.read(key) for key in complete}
        hidden = {key: cache.outputs[key] for key in complete}
        computed = policy['zero']
        torch.cuda.synchronize(backend.device)
        started = time.perf_counter()
        if cold:
            ids, mask = padded([list(key) for key in cold], backend.tokenizer.pad_token_id,
                               backend.device, 'left')
            transport.before_forward()
            output = transport.trunk(inputs_embeds=transport.get_input_embeddings()(ids),
                attention_mask=mask, position_ids=(mask.cumsum(-1) - policy['step']).clamp_min(policy['zero']),
                past_key_values=DynamicCache(config=backend.cache_config), use_cache=True,
                output_hidden_states=True, return_dict=True)
            snapshots = split_native_cache(output.past_key_values, list(map(len, cold)))
            terminal = torch.stack([layer[:, policy['last_index']] for layer in output.hidden_states], dim=policy['layer_axis'])
            states.update(zip(cold, snapshots, strict=True))
            hidden.update((key, terminal[row:row + policy['step']].clone()) for row, key in enumerate(cold))
            computed += sum(map(len, cold))
        if partial:
            before = [list(key[:policy['last_index']]) for key in partial]
            snapshots = self.history.compute(before)
            packed, mask = pack_native_caches(snapshots)
            lengths = torch.tensor(list(map(len, before)), device=backend.device)
            ids = torch.tensor([[key[policy['last_index']]] for key in partial], device=backend.device)
            mask = torch.cat([mask, torch.ones_like(ids, dtype=mask.dtype)], dim=policy['sequence_axis'])
            transport.before_forward()
            output = transport.trunk(inputs_embeds=transport.get_input_embeddings()(ids),
                attention_mask=mask, position_ids=lengths[:, None], past_key_values=packed,
                use_cache=True, output_hidden_states=True, return_dict=True)
            snapshots = split_native_cache(output.past_key_values, list(map(len, partial)))
            terminal = torch.stack([layer[:, policy['last_index']] for layer in output.hidden_states], dim=policy['layer_axis'])
            states.update(zip(partial, snapshots, strict=True))
            hidden.update((key, terminal[row:row + policy['step']].clone()) for row, key in enumerate(partial))
            computed += self.history.records[policy['last_index']]['computed_input_tokens'] + len(partial)
        for key in cold + partial:
            cache.store(key, states[key], hidden[key])
        decoder = transport.decoder
        load_native_prefixes(decoder, [states[key] for key in sequences],
                             [key[policy['last_index']] for key in sequences], policy['history']['state_copy'])
        width = int(decoder.cache.get_seq_length())
        transport.cache = decoder.cache
        transport.mask = decoder.key_valid[:, :width].clone()
        terminal = torch.cat([hidden[key] for key in sequences], dim=policy['batch_axis'])
        layers = tuple(value[:, None] for value in terminal.unbind(dim=policy['layer_axis']))
        transport.last_hidden = terminal[:, policy['last_index']]
        torch.cuda.synchronize(backend.device)
        self.records.append({'source_identity': policy['source_identity'], 'batch_size': len(sequences),
            'logical_input_tokens': sum(map(len, sequences)), 'computed_input_tokens': computed,
            'complete_hits': len(complete), 'cold_rows': len(cold), 'partial_rows': len(partial),
            'cache_bytes': cache.bytes, 'elapsed_seconds': time.perf_counter() - started})
        return BaseModelOutputWithPast(last_hidden_state=layers[policy['last_index']],
                                       past_key_values=decoder.cache, hidden_states=layers)


class CachedRoleTransport(CapturedPaddingTransport):
    def __init__(self, *args, history, sequences, **kwargs):
        super().__init__(*args, **kwargs)
        self.history, self.sequences = history, sequences

    def __call__(self, input_ids=None, inputs_embeds=None, attention_mask=None,
                 past_key_values=None, **kwargs):
        if past_key_values is None:
            assert input_ids is not None and inputs_embeds is None
            return self.history.prefill(self, self.sequences)
        return super().__call__(input_ids=input_ids, inputs_embeds=inputs_embeds,
                                attention_mask=attention_mask, past_key_values=past_key_values, **kwargs)
