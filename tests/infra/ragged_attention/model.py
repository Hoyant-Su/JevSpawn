import argparse
import json
from pathlib import Path
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F

from baselines.common.config import SharedConfig
from baselines.common.parallel_run import initialize_parallel
from jev_spawn.algo.structured import padded
from jev_spawn.infra.cached_attention import RaggedCacheAttention
from jev_spawn.runtime.native_cache_batch import pack_native_caches, split_native_cache
from jev_spawn.runtime.ragged_suffix import RaggedSuffix
from jev_spawn.schema import CONTROLLER, controller_prompts


@torch.inference_mode()
def main(settings):
    shared = SharedConfig.load(settings['shared_config'])
    backend, commands, startup = initialize_parallel(shared, json.loads(Path(settings['parallel_settings']).read_text()))
    recorded = json.loads(Path(settings['requests']).read_text())
    assert len(recorded) == shared.runtime.batch_size
    requests = [item['request'] for item in recorded]
    prompts = [controller_prompts([request['state']], request['question'], request['options'],
                list(backend.answer_labels[:len(request['options'])]), CONTROLLER['output_instruction'],
                contexts=[request['context']])[0] for request in requests]
    text = backend._render(prompts, CONTROLLER['system'])
    sequences = backend.tokenizer(text, add_special_tokens=False)['input_ids']
    assert max(map(len, sequences)) <= shared.model.max_input_tokens
    splits = [len(sequence) // settings['split_divisor'] for sequence in sequences]
    prefixes = [sequence[:cut] for sequence, cut in zip(sequences, splits, strict=True)]
    tails = [sequence[cut:] for sequence, cut in zip(sequences, splits, strict=True)]
    ids, mask = padded(prefixes, backend.tokenizer.pad_token_id, backend.device, 'left')
    roots = backend.model.model(input_ids=ids, attention_mask=mask,
        position_ids=(mask.cumsum(-1) - 1).clamp_min(0), use_cache=True)
    states = split_native_cache(roots.past_key_values, splits)
    del roots, ids, mask
    outputs, measurements = {}, {}
    for name in ('sdpa', 'varlen'):
        cache, prefix_mask = pack_native_caches(states)
        ids, mask = padded(tails, backend.tokenizer.pad_token_id, backend.device, 'right')
        full_mask = torch.cat([prefix_mask, mask], dim=-1)
        positions = (full_mask.cumsum(-1) - 1).clamp_min(0)[:, -ids.shape[1]:]
        descriptor = RaggedSuffix(list(map(len, tails)), backend.device)
        attention = None if name == 'sdpa' else RaggedCacheAttention(descriptor, splits)
        torch.cuda.synchronize(backend.device)
        torch.cuda.reset_peak_memory_stats(backend.device)
        baseline = torch.cuda.memory_allocated(backend.device)
        start = time.perf_counter()
        result = backend.model.model(input_ids=ids, attention_mask=full_mask, position_ids=positions,
            past_key_values=cache, use_cache=True, ragged_suffix=descriptor, decode_attention=attention)
        ends = torch.tensor(list(map(len, tails)), device=backend.device) - 1
        hidden = result.last_hidden_state[torch.arange(len(tails), device=backend.device), ends]
        outputs[name] = F.linear(hidden.float(), backend.finite_output_weights)
        torch.cuda.synchronize(backend.device)
        measurements[name] = {'seconds': time.perf_counter() - start,
            'extra_peak_bytes': torch.cuda.max_memory_allocated(backend.device) - baseline}
        del result, cache, ids, mask, full_mask, hidden, descriptor, attention
    rows = []
    for row, request in enumerate(requests):
        count = len(request['options'])
        original, packed = outputs['sdpa'][row, :count], outputs['varlen'][row, :count]
        rows.append({'source': recorded[row]['source'], 'turn': recorded[row]['turn'],
            'input_tokens': len(sequences[row]), 'prefix_tokens': splits[row], 'options': count,
            'max_logit_error': (original - packed).abs().max().item(),
            'choice_agreement': bool(original.argmax() == packed.argmax()),
            'sdpa_logits': original.tolist(), 'varlen_logits': packed.tolist()})
    report = {'scope': 'Paired full 27B model continuation on eight real recorded finite requests; not task accuracy.',
              'startup': startup, 'batch_size': len(rows), 'measurements': measurements, 'rows': rows}
    output = Path(settings['output'])
    output.mkdir(parents=True, exist_ok=True)
    (output / f'rank-{dist.get_rank()}.json').write_text(json.dumps(report, indent=2) + '\n')
    assert max(row['max_logit_error'] for row in rows) <= settings['logit_max_absolute_error']
    assert sum(row['choice_agreement'] for row in rows) / len(rows) >= settings['required_choice_agreement']
    print(json.dumps({'rank': dist.get_rank(), 'measurements': measurements, 'rows': rows}), flush=True)
    dist.destroy_process_group(commands.control_group)
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    main(json.loads(parser.parse_args().config.read_text()))
