import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import time

import torch
import torch.distributed as dist

from baselines.common.config import SharedConfig
from baselines.common.parallel_run import initialize_parallel
from baselines.official.model_service import GenerationService, RowStops


@torch.inference_mode()
def run(config):
    shared = SharedConfig.load(config['shared_config'])
    parallel = json.loads(Path(config['parallel_settings']).read_text())
    backend, commands, startup = initialize_parallel(shared, parallel)
    batches = json.loads(Path(config['source_batches']).read_text())
    batches = [batch for batch in batches if batch.get('messages')]
    results = []
    for index in config['generation_cohorts']:
        source = batches[index]
        assert source['temperature'] == shared.generation.temperature == 0
        assert len(source['messages']) == source['batch_size']
        rendered = backend.tokenizer.apply_chat_template(source['messages'], tokenize=False,
            add_generation_prompt=True, enable_thinking=shared.generation.enable_thinking)
        inputs = backend.tokenizer(rendered, padding=True, truncation=False,
            add_special_tokens=False, return_tensors='pt').to(backend.device)
        lengths = inputs['attention_mask'].sum(-1).tolist()
        assert lengths == source['input_tokens']
        assert max(lengths) <= shared.model.max_input_tokens
        assert len(set(source['requested_max_new_tokens'])) == 1
        cap = source['max_new_tokens']
        assert cap <= shared.generation.max_new_tokens
        width = inputs['input_ids'].shape[-1]
        stopping = RowStops(backend.tokenizer, source['stop'], backend.eos_ids,
                            width, source['batch_size'], backend.device)
        options = {'do_sample': False, 'max_new_tokens': cap, 'use_cache': True,
            'eos_token_id': backend.eos_ids, 'pad_token_id': backend.tokenizer.pad_token_id,
            'bos_token_id': backend.tokenizer.bos_token_id}
        torch.cuda.synchronize()
        started = time.perf_counter()
        sequences, events = GenerationService._generate_tokens(SimpleNamespace(backend=backend),
            inputs, options, stopping)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        generated = sequences[:, width:].tolist()
        counts = stopping.lengths.tolist()
        tokens = [row[:count or len(row)] for row, count in zip(generated, counts, strict=True)]
        texts = backend.tokenizer.batch_decode(tokens, skip_special_tokens=True)
        results.append({'generation_cohort': index, 'task_ids': source['task_ids'],
            'batch_size': source['batch_size'], 'input_tokens': lengths, 'max_new_tokens': cap,
            'recorded_graph_texts': source['texts'], 'native_eager_texts': texts,
            'recorded_graph_token_ids': source['output_token_ids'], 'native_eager_token_ids': tokens,
            'same_tokens': [left == right for left, right in zip(tokens, source['output_token_ids'], strict=True)],
            'native_seconds': elapsed, 'native_forward_steps': len(events)})
        if commands.is_leader:
            Path(config['output']).write_text(json.dumps({'config': config, 'startup': startup,
                'status': 'running', 'records': results}, indent=2) + '\n')
            print(json.dumps({'cohort': index, 'same_tokens': results[-1]['same_tokens'],
                'native_eager_texts': texts, 'native_seconds': elapsed}), flush=True)
    if commands.is_leader:
        Path(config['output']).write_text(json.dumps({'config': config, 'startup': startup,
            'status': 'complete', 'records': results}, indent=2) + '\n')
    dist.destroy_process_group(commands.control_group)
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    run(json.loads(parser.parse_args().config.read_text()))
