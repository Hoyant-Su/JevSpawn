import time
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from transformers.cache_utils import LinearAttentionCacheLayerMixin

from baselines.latentmas.adapter import ModelAdapter
from methods.latent_readout.inputs import candidate_ids, render


def state_shapes(transport):
    layers = []
    for layer in transport.cache.layers:
        if isinstance(layer, LinearAttentionCacheLayerMixin):
            shapes = {'conv': list(layer.conv_states[0].shape),
                      'recurrent': list(layer.recurrent_states[0].shape)}
        else:
            shapes = {'keys': list(layer.keys.shape), 'values': list(layer.values.shape)}
        layers.append({'type': type(layer).__name__, **shapes})
    return layers


class ReadoutExperiment:
    def __init__(self, backend, settings, prompts):
        self.backend, self.settings, self.prompts = backend, settings, prompts
        self.wrapper = ModelAdapter(backend, SimpleNamespace(latent_space_realign=True))

    @torch.inference_mode()
    def measure(self, tasks, condition):
        backend, wrapper = self.backend, self.wrapper
        wrapper.reset(capture=False)
        torch.manual_seed(self.settings['seed'])
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.perf_counter()
        assert backend.tokenizer.padding_side == 'left'
        assert backend.model.get_input_embeddings().weight.dtype == torch.bfloat16
        assert backend.model.get_input_embeddings().weight.device == backend.device
        encoded, lengths = backend._encode(render(backend.tokenizer, tasks, self.prompts))
        mask = encoded['attention_mask']
        assert encoded['input_ids'].dtype == torch.long and encoded['input_ids'].device == backend.device
        assert bool((mask[:, -1] == 1).all())
        assert bool(((mask == 0) | (mask == 1)).all())
        assert bool((mask[:, 1:] >= mask[:, :-1]).all())
        batch_size = len(tasks)
        steps = condition['steps'] if condition['mode'] == 'latent' else 0
        wrapper.generate_latent_batch(encoded['input_ids'], encoded['attention_mask'],
                                     latent_steps=steps, past_key_values=None)
        transport = wrapper.model
        assert transport.last_hidden.dtype == torch.bfloat16 and transport.last_hidden.device == backend.device
        assert len(transport.records) == steps + 1
        phases = {'prefill_trunk': transport.records[0]['seconds'],
                  'latent_trunk': sum(row['seconds'] for row in transport.records[1:])}
        transport.records[0]['phase'] = 'prefill'
        for row in transport.records[1:]:
            row['phase'] = 'latent'
        generated, token_times = [], []
        finished = torch.ones(batch_size, dtype=torch.bool, device=backend.device)
        if condition['mode'] == 'text':
            assert condition['temperature'] == 0 and condition['require_eos']
            finished.zero_()
            eos = torch.tensor(backend.eos_ids, device=backend.device)
            transport.phase = 'text'
            text_start = time.perf_counter()
            for _ in range(condition['max_new_tokens']):
                next_ids = backend.model.lm_head(transport.last_hidden).argmax(-1)
                next_ids = torch.where(finished, backend.tokenizer.pad_token_id, next_ids)
                generated.append(next_ids)
                torch.cuda.synchronize()
                token_times.append(time.perf_counter())
                finished |= torch.isin(next_ids, eos)
                if bool(finished.all()):
                    break
                # EOS is selected but never fed. Every actual nonterminal token is fed,
                # including the final budgeted token when no EOS was produced.
                transport(input_ids=torch.where(finished, backend.tokenizer.pad_token_id, next_ids)[:, None],
                          attention_mask=(~finished).long()[:, None], past_key_values=transport.cache)
            phases['text_reasoning'] = time.perf_counter() - text_start
        else:
            phases['text_reasoning'] = 0.0
        valid = finished.tolist()
        sequences = torch.stack(generated, dim=1).tolist() if generated else [[] for _ in tasks]
        reasoning, selected_eos, intervals = [], [], []
        for sequence in sequences:
            stop = next((index for index, token in enumerate(sequence) if token in backend.eos_ids), len(sequence))
            reasoning.append(sequence[:stop])
            selected_eos.append(sequence[stop] if stop < len(sequence) else None)
            count = stop + (stop < len(sequence))
            intervals.append([token_times[index] - token_times[index - 1] for index in range(1, count)])
        suffix_start = time.perf_counter()
        suffix = backend.tokenizer.encode(self.settings['readout_suffix'], add_special_tokens=False)
        transport.phase = 'suffix'
        readout_hidden = transport.last_hidden
        if any(valid):
            for token in suffix:
                output = transport(input_ids=torch.full((batch_size, 1), token, device=backend.device),
                                   attention_mask=finished.long()[:, None], past_key_values=transport.cache)
            readout_hidden = output.last_hidden_state[:, -1]
            assert transport.mask[:, -1].bool().tolist() == valid
            assert torch.equal(readout_hidden[finished], transport.last_hidden[finished])
        assert readout_hidden.dtype == torch.bfloat16 and readout_hidden.device == backend.device
        torch.cuda.synchronize()
        phases['suffix'] = time.perf_counter() - suffix_start
        readout_start = time.perf_counter()
        counts = [len(task['field']['options']) for task in tasks]
        ids = candidate_ids(backend.tokenizer, max(counts), self.settings['readout_suffix'])
        head = backend.model.lm_head
        indices = torch.tensor(ids, device=backend.device)
        weight = head.weight.index_select(0, indices).float()
        bias = None if head.bias is None else head.bias.index_select(0, indices).float()
        logits = F.linear(readout_hidden.float(), weight, bias)
        allowed = torch.arange(len(ids), device=backend.device)[None, :] < torch.tensor(counts, device=backend.device)[:, None]
        probabilities = logits.masked_fill(~allowed, -torch.inf).softmax(-1)
        choices = probabilities.argmax(-1).tolist()
        logits, probabilities = logits.tolist(), probabilities.tolist()
        torch.cuda.synchronize()
        phases['readout'] = time.perf_counter() - readout_start
        rows = []
        decoded = backend.tokenizer.batch_decode(reasoning, skip_special_tokens=True)
        for index, task in enumerate(tasks):
            options = task['field']['options']
            rows.append({'task_id': task['task_id'], 'dataset': task['dataset'], 'field_id': task['field_id'],
                         'valid': valid[index], 'answer': options[choices[index]]['id'] if valid[index] else None,
                         'option_ids': [option['id'] for option in options],
                         'logits': logits[index][:counts[index]] if valid[index] else None,
                         'probabilities': probabilities[index][:counts[index]] if valid[index] else None,
                         'input_tokens': lengths[index], 'latent_steps': steps,
                         'reasoning_token_ids': reasoning[index], 'selected_eos': selected_eos[index],
                         'reasoning': decoded[index], 'generated_tokens': len(reasoning[index]) + (selected_eos[index] is not None),
                         'text_itl_seconds': intervals[index]})
        expected = [length + steps + len(tokens) + len(suffix) * is_valid
                    for length, tokens, is_valid in zip(lengths, reasoning, valid)]
        assert transport.mask.sum(1).tolist() == expected
        assert all(row['batch_size'] == batch_size for row in transport.records)
        result = {'condition': condition, 'batch_size': batch_size, 'predictions': rows,
                  'phase_seconds': phases, 'forward_calls': transport.records,
                  'physical_history_tokens': transport.mask.shape[1], 'valid_history_tokens': expected,
                  'cache_shapes': state_shapes(transport), 'restored_padding_rows': transport.restored_rows,
                  'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                  'peak_reserved_bytes': torch.cuda.max_memory_reserved()}
        torch.cuda.synchronize()
        result['compute_seconds'] = time.perf_counter() - start
        phases['other_compute'] = result['compute_seconds'] - sum(phases.values())
        assert phases['other_compute'] >= 0
        return result
