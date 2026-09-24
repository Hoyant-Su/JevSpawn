import argparse
import json
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F

from baselines.common.config import SharedConfig
from baselines.common.parallel_run import initialize_parallel
from jev_spawn.algo.structured import padded
from jev_spawn.infra.kernel_tuning import load_tuning
from jev_spawn.runtime.cache_arena import StaticCacheArena
from jev_spawn.runtime.decoding import CapturedDecode


def run(config):
    shared = SharedConfig.load(config['shared_config'])
    backend, commands, startup = initialize_parallel(shared, json.loads(Path(config['parallel_settings']).read_text()))
    tuning = load_tuning(Path(config['cache_directory']) / config['cache_file'].format(rank=dist.get_rank()),
                        backend, json.loads(Path(config['tuning_settings']).read_text()))
    output = Path(config['output'])
    output.mkdir(parents=True, exist_ok=True)

    @torch.inference_mode()
    def compute(records):
        sequences = [record['tokens'] for record in records]
        counts = [len(record['field']['options']) for record in records]
        native_ids = tuple(backend.answer_label_ids[:max(counts)])
        selected_weights = backend.selected_output_weights(native_ids).float()
        ids, mask = padded(sequences, backend.tokenizer.pad_token_id, backend.device, 'left')
        capacity = ids.shape[-1] + config['generation_tokens']
        arena = StaticCacheArena(backend.cache_config, len(records), capacity,
                                 backend.model.lm_head.weight.dtype, backend.device)
        decoder = CapturedDecode(backend, len(records), capacity, arena=arena)
        first = {}

        def capture_hidden(module, arguments, result):
            first['hidden'] = result.last_hidden_state[:, -1].clone()
            hidden_hook.remove()

        def capture_logits(module, arguments, result):
            first['logits'] = result.clone()
            logits_hook.remove()

        hidden_hook = decoder.trunk.register_forward_hook(capture_hidden)
        logits_hook = backend.model.lm_head.register_forward_hook(capture_logits)
        eos = torch.tensor(backend.eos_ids, device=backend.device)

        def stopping(tokens, scores):
            return torch.isin(tokens[:, -1], eos)

        options = {'max_new_tokens': config['generation_tokens'], 'pad_token_id': backend.tokenizer.pad_token_id,
                   'do_sample': config['do_sample'], 'temperature': shared.generation.temperature}
        generated, events, capture_seconds = decoder.generate_tokens(
            {'input_ids': ids, 'attention_mask': mask}, options, stopping, shared.runtime.graph_warmup_steps)
        selected_logits = F.linear(first['hidden'].float(), selected_weights)
        native_logits = first['logits'].float()
        probabilities = native_logits.softmax(-1)
        top_logits, top_ids = native_logits.topk(config['top_predictions'], dim=-1)
        top_probabilities = probabilities.gather(-1, top_ids)
        valid_ids = torch.tensor(native_ids, device=backend.device)
        native_selected_logits = native_logits.index_select(-1, valid_ids)
        native_label_mass = probabilities.index_select(-1, valid_ids)
        generated = generated[:, ids.shape[-1]:].tolist()
        rows = []
        for index, (record, count, tokens) in enumerate(zip(records, counts, generated, strict=True)):
            stop = next((offset + 1 for offset, token in enumerate(tokens) if token in backend.eos_ids), len(tokens))
            tokens = tokens[:stop]
            logits = selected_logits[index, :count].tolist()
            forced = selected_logits[index, :count].argmax().item()
            native_forced = native_selected_logits[index, :count].argmax().item()
            top = top_ids[index].tolist()
            rows.append({'task_id': record['task_id'], 'node': record['field']['id'],
                'input_tokens': len(record['tokens']), 'option_ids': [item['id'] for item in record['field']['options']],
                'native_label_ids': list(native_ids[:count]), 'native_labels': backend.answer_labels[:count],
                'finite_selected_logits_fp32': logits,
                'finite_choice_fp32': record['field']['options'][forced]['id'],
                'native_head_selected_logits': native_selected_logits[index, :count].tolist(),
                'finite_choice_native_head': record['field']['options'][native_forced]['id'],
                'valid_label_mass_native_head': native_label_mass[index, :count].sum().item(),
                'top_native_predictions': [{'id': token, 'text': backend.tokenizer.decode([token]),
                    'logit': logit, 'probability': probability} for token, logit, probability in
                    zip(top, top_logits[index].tolist(), top_probabilities[index].tolist(), strict=True)],
                'generated_token_ids': tokens, 'generated_token_texts': [backend.tokenizer.decode([token]) for token in tokens],
                'generated_text': backend.tokenizer.decode(tokens, skip_special_tokens=True),
                'ended_with_eos': tokens[-1] in backend.eos_ids,
                'greedy_first_token_is_valid_label': tokens[0] in native_ids[:count],
                'recorded_forced_choice': record['recorded_result']['choice']})
        ranks = [None for _ in range(shared.runtime.world_size)]
        dist.all_gather_object(ranks, rows, group=commands.control_group)
        assert all(rank == rows for rank in ranks)
        report = {'rows': rows, 'all_rank_outputs_equal': True, 'actual_batch_size': len(records),
                  'padded_input_shape': list(ids.shape), 'generation_token_cap': config['generation_tokens'],
                  'graph_capture_seconds': capture_seconds, 'shared_config': config['shared_config'],
                  'scope': config['scope']}
        (output / config['rank_file'].format(rank=dist.get_rank())).write_text(json.dumps(report, indent=2) + '\n')
        decoder.graph = None
        return report

    commands.register(config['command'], compute)
    if commands.is_leader:
        try:
            source = Path(config['source_run'])
            task = next(json.loads(path.read_text()) for path in source.glob('task-*.json')
                        if json.loads(path.read_text())['task_id'] == config['task_id'])
            calls = {call['node']: call['result'] for call in task['calls'] if call['kind'] == 'finite'}
            records = [record for path in sorted(source.glob(config['input_glob']))
                       for record in json.loads(path.read_text())['requests']
                       if record['task_id'] == config['task_id'] and any(
                           isinstance(value, dict) and config['domain_marker'] in value
                           for value in record['field'].get('candidate_values', []))]
            assert len(records) == config['expected_requests']
            for record in records:
                tokens = backend.tokenizer(record['rendered'], add_special_tokens=False)['input_ids']
                assert tokens == record['tokens'] and len(tokens) == record['input_tokens']
                assert len(tokens) <= shared.model.max_input_tokens
                record['recorded_result'] = calls[record['field']['id']]
            (output / config['inputs_file']).write_text(json.dumps(records, ensure_ascii=False, indent=2) + '\n')
            report = commands.call(config['command'], records)
            Path(config['result']).write_text(json.dumps({'config': config, 'startup': startup,
                'tuning': tuning, 'report': report}, indent=2) + '\n')
        finally:
            commands.finish()
    else:
        commands.serve()
    dist.destroy_process_group(commands.control_group)
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    run(json.loads(Path(parser.parse_args().config).read_text()))
