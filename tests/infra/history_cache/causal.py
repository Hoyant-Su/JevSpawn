import argparse
import gc
import json
from pathlib import Path

import torch
import torch.distributed as dist
from transformers.cache_utils import StaticLayer

from baselines.common.config import SharedConfig
from baselines.common.parallel_run import initialize_parallel
from jev_spawn.algo.structured import padded
from jev_spawn.infra.kernel_tuning import load_tuning
from jev_spawn.runtime.cache_arena import StaticCacheArena
from jev_spawn.runtime.decoding import CapturedDecode
from jev_spawn.runtime.native_cache_batch import split_native_cache
from jev_spawn.runtime.native_restore import load_native_prefixes


def compare(left, right):
    delta = (left.float() - right.float()).abs()
    return {'exact': torch.equal(left, right), 'max_abs': float(delta.max()),
            'mean_abs': float(delta.mean()), 'reference_max_abs': float(left.float().abs().max())}


def run(config):
    shared = SharedConfig.load(config['shared_config'])
    backend, commands, startup = initialize_parallel(shared, json.loads(Path(config['parallel_settings']).read_text()))
    load_tuning(Path(config['cache_directory']) / config['cache_file'].format(rank=dist.get_rank()), backend,
                json.loads(Path(config['tuning_settings']).read_text()))
    source = json.loads(Path(config['source']).read_text())
    output = Path(config['output'])
    output.mkdir(parents=True, exist_ok=True)
    policy = json.loads(Path(config['history_settings']).read_text())
    trunk = backend.model.model.language_model

    @torch.inference_mode()
    def compute(workload):
        batch = source[workload['batch']]
        texts = backend.tokenizer.apply_chat_template([batch['messages'][row] for row in workload['rows']],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        sequences = backend.tokenizer(texts, add_special_tokens=False, truncation=False)['input_ids']
        lengths = list(map(len, sequences))
        assert lengths == [batch['input_tokens'][row] for row in workload['rows']]
        ids, mask = padded(sequences, backend.tokenizer.pad_token_id, backend.device, 'left')
        positions = (mask.cumsum(-1) - policy['position_step']).clamp_min(policy['zero'])
        records, hidden, logits, decoders = {}, {}, {}, []

        def observed(name, call):
            values = []
            hooks = [layer.register_forward_hook(lambda module, arguments, result:
                values.append(result[:, -1].clone())) for layer in trunk.layers]
            result = call()
            for hook in hooks:
                hook.remove()
            hidden[name] = values
            return result

        def decoder(capacity):
            arena = StaticCacheArena(backend.cache_config, len(sequences), capacity,
                backend.model.lm_head.weight.dtype, backend.device)
            value = CapturedDecode(backend, len(sequences), capacity, arena=arena)
            decoders.append(value)
            return value

        full = decoder(shared.model.max_input_tokens + config['generation_tokens'])
        observed('full_static', lambda: full.prefill({'input_ids': ids, 'attention_mask': mask}))
        logits['full_static'] = full.logits.clone()
        dynamic = observed('full_dynamic', lambda: trunk(input_ids=ids, attention_mask=mask,
            position_ids=positions, use_cache=True))
        logits['full_dynamic'] = backend.model.lm_head(dynamic.last_hidden_state[:, -1]).clone()
        del dynamic
        prefix_ids, prefix_mask = padded([sequence[:-1] for sequence in sequences],
            backend.tokenizer.pad_token_id, backend.device, 'left')
        prefix = trunk(input_ids=prefix_ids, attention_mask=prefix_mask,
            position_ids=(prefix_mask.cumsum(-1) - policy['position_step']).clamp_min(policy['zero']), use_cache=True)
        states = split_native_cache(prefix.past_key_values, [length - policy['position_step'] for length in lengths])
        last = torch.tensor([sequence[-1] for sequence in sequences], device=backend.device)[:, None]
        split = observed('split_dynamic', lambda: trunk(input_ids=last, attention_mask=mask,
            position_ids=positions[:, -1:], past_key_values=prefix.past_key_values, use_cache=True))
        logits['split_dynamic'] = backend.model.lm_head(split.last_hidden_state[:, -1]).clone()
        del split, prefix
        for mode in config['restore_modes']:
            restored = decoder(ids.shape[-1])
            load_native_prefixes(restored, states, last[:, 0].tolist(), policy['state_copy'])
            copies = []
            for row, state in enumerate(states):
                length = state.get_seq_length()
                width = int(restored.cache.get_seq_length())
                for layer, target in zip(state.layers, restored.cache.layers, strict=True):
                    if type(target) is StaticLayer:
                        copies.extend(torch.equal(getattr(layer, name), getattr(target, name)[row:row+1, :, width-length:width])
                                      for name in ('keys','values'))
                    else:
                        copies.extend(torch.equal(tensor, getattr(target, name)[key][row:row+1])
                            for name in ('conv_states','recurrent_states') for key,tensor in getattr(layer,name).items())
            assert all(copies)
            records[mode] = {'restore_tensor_copies_exact': all(copies),
                'positions': restored.positions[:, 0].tolist(), 'left_padding': restored.leftpad.tolist()}
            if mode == config['native_restore_mode']:
                restored.decode_kwargs = {}
            observed(mode, restored.step)
            logits[mode] = restored.logits.clone()
        comparisons = {}
        for left, right in config['comparisons']:
            comparisons[left + '__' + right] = {'logits': compare(logits[left], logits[right]),
                'layers': [dict(layer=index, type=trunk.layers[index].block_type, **compare(a,b))
                           for index,(a,b) in enumerate(zip(hidden[left],hidden[right],strict=True))]}
        selected = {name: {'tokens': value.argmax(-1).tolist(),
            'top_ids': value.topk(config['top_count'], dim=-1).indices.tolist(),
            'top_logits': value.topk(config['top_count'], dim=-1).values.float().tolist()}
            for name,value in logits.items()}
        gathered = [None] * shared.runtime.world_size
        dist.all_gather_object(gathered, selected, group=commands.control_group)
        assert all(item == selected for item in gathered)
        result = {'name':workload['name'],'input_tokens':lengths,'source_rows':workload['rows'],
                  'allrank_equal':True,'outputs':selected,'restore':records,'comparisons':comparisons}
        (output / config['rank_file'].format(name=workload['name'],rank=dist.get_rank())).write_text(json.dumps(result,indent=2)+'\n')
        return result

    commands.register(config['command'],compute)
    if commands.is_leader:
        try:
            reports=[commands.call(config['command'],workload) for workload in config['workloads']]
            Path(config['result']).write_text(json.dumps({'config':config,'reports':reports},indent=2)+'\n')
        finally:
            commands.finish()
    else:
        commands.serve()
    gc.collect()
    dist.destroy_process_group(commands.control_group)
    dist.destroy_process_group()


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    run(json.loads(Path(parser.parse_args().config).read_text()))
