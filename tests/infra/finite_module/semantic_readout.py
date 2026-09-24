import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F

from baselines.common.config import SharedConfig
from baselines.common.parallel_run import initialize_parallel
from jev_spawn.algo.structured import padded
from jev_spawn.infra.semantic_domain import compile_domain


def read(path):
    return json.loads(Path(path).read_text())


@torch.inference_mode()
def run(config):
    shared = SharedConfig.load(config['shared_config'])
    backend, commands, startup = initialize_parallel(shared, read(config['parallel_settings']))
    domain = compile_domain(backend.tokenizer, config['values'])
    weights = backend.selected_output_weights(domain.branches).float()
    sources = [read(Path(config['source']) / config['trace_file'].format(index=index))
               for index in range(config['task_count'])]
    sequences = []
    for source in sources:
        rendered = backend.tokenizer.apply_chat_template(source['messages'], tokenize=False,
            add_generation_prompt=True, enable_thinking=shared.generation.enable_thinking)
        tokens = backend.tokenizer(rendered, add_special_tokens=False)['input_ids']
        for value, tail in zip(domain.values, domain.tails, strict=True):
            assert backend.tokenizer(rendered + value, add_special_tokens=False)['input_ids'] == (
                tokens + list(domain.prefix) + list(tail))
        sequences.append(tokens + list(domain.prefix))
    records = []
    for offset in range(0, len(sequences), config['readout_batch_size']):
        batch = sequences[offset:offset + config['readout_batch_size']]
        ids, mask = padded(batch, backend.tokenizer.pad_token_id, backend.device, config['padding'])
        torch.cuda.synchronize(backend.device)
        started = time.perf_counter()
        output = backend.model.model(input_ids=ids, attention_mask=mask,
            position_ids=(mask.cumsum(-1) - 1).clamp_min(0), use_cache=False)
        logits = F.linear(output.last_hidden_state[:, -1].float(), weights)
        choices = logits.argmax(-1).tolist()
        torch.cuda.synchronize(backend.device)
        elapsed = time.perf_counter() - started
        records.append({'offset': offset, 'batch_size': len(batch), 'input_tokens': list(map(len, batch)),
            'seconds': elapsed, 'choices': [domain.values[index] for index in choices],
            'logits': logits.tolist()})
        if commands.is_leader:
            destination = Path(config['output'])
            destination.mkdir(parents=True, exist_ok=True)
            (destination / 'results.json').write_text(json.dumps(
                {'config': config, 'domain': asdict(domain), 'startup': startup, 'records': records}, indent=2)+'\n')
            print(json.dumps(records[-1]), flush=True)
    dist.destroy_process_group(commands.control_group)
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    run(read(parser.parse_args().config))
