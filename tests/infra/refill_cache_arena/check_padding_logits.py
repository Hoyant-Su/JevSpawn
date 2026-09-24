import argparse
import json
from pathlib import Path

import torch
import torch.distributed as dist

from baselines.common.config import SharedConfig
from baselines.common.parallel_run import initialize_parallel
from jev_spawn.runtime.decoding import CapturedDecode
from jev_spawn.runtime.rolling_decode import RollingDecode
from jev_spawn.runtime.padded_rolling_decode import PaddedRollingDecode


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--specification', type=Path, required=True)
    args = parser.parse_args()
    settings = json.loads(args.specification.read_text())
    shared = SharedConfig.load(settings['shared_config'])
    backend, commands, starts = initialize_parallel(shared, json.loads(Path(settings['parallel_settings']).read_text()))
    batches = json.loads(Path(settings['source_batches']).read_text())
    batch = next(row for row in batches if settings['request_id'] in row['task_ids'])
    row = batch['task_ids'].index(settings['request_id'])
    history = batch['output_token_ids'][row]
    rendered = backend.tokenizer.apply_chat_template(batch['messages'][row], tokenize=False,
                    add_generation_prompt=True, enable_thinking=False)
    prompt = backend.tokenizer(rendered, add_special_tokens=False)['input_ids']
    prior = settings['first_difference_index']
    context = prompt + history[:prior - 1]
    ids = torch.tensor([context], device=backend.device, dtype=torch.long)
    source = CapturedDecode(backend, len(ids), settings['capacity'])
    source.prefill({'input_ids': ids, 'attention_mask': torch.ones_like(ids)})
    source.ids.fill_(history[prior - 1])
    logits = {}
    for arm in settings['arms']:
        cls = PaddedRollingDecode if arm['padding'] else RollingDecode
        decoder = cls(backend, arm['physical_rows'], settings['capacity'], source)
        indices = torch.zeros(arm['physical_rows'], device=backend.device, dtype=torch.long)
        decoder.load([(source, indices)])
        if arm['padding']:
            decoder.configure_rows(settings['active_rows'])
        decoder.step()
        logits[arm['name']] = decoder.logits[settings['target_row']].float().clone()
    if commands.is_leader:
        stats = {}
        for name, values in logits.items():
            scores, indices = values.topk(settings['topk'])
            stats[name] = {'argmax': int(indices[0]), 'top_token_ids': indices.tolist(),
                'top_logits': scores.tolist(), 'top_margin': float(scores[0] - scores[1]),
                'observed_reference_token_logit': float(values[settings['reference_token']]),
                'observed_candidate_token_logit': float(values[settings['candidate_token']])}
        comparisons = {}
        for pair in settings['comparisons']:
            a, b = (logits[name] for name in pair)
            comparisons[' versus '.join(pair)] = {'max_absolute_logit_difference': float((a-b).abs().max()),
                'rms_logit_difference': float((a-b).square().mean().sqrt()), 'exact_logits': torch.equal(a,b)}
        result = {'settings': settings, 'shared_observed_prior_tokens': prior,
            'actual_conditioned_context_tokens': len(context)+1, 'arms': stats, 'comparisons': comparisons,
            'scope': 'Same real observed prefix and exact copied starting cache on every arm. OrdinaryB1/B8 versus active1+paddedB8 next-token operator check, not a new task run or forced output.'}
        Path(settings['output']).write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result), flush=True)
    dist.barrier(group=commands.control_group)
    dist.destroy_process_group(commands.control_group)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
