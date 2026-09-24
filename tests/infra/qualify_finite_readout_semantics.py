import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import GenerationConfig

from baselines.common.config import SharedConfig
from jev_spawn.infra.backend import Backend
from jev_spawn.infra.configuration import CORE
from jev_spawn.infra.prompts import load_prompt
from jev_spawn.infra.readout_labels import validate_boundaries
from jev_spawn.schema import CONTROLLER
from qualify_finite_prefix_independent import render, save, workload


def token_details(tokenizer, ids):
    return [{'token_id': identity, 'token': tokenizer.convert_ids_to_tokens(identity),
             'decoded': tokenizer.decode([identity], skip_special_tokens=False)} for identity in ids]


@torch.inference_mode()
def run(settings):
    replay = json.loads(Path(settings['replay_specification']).read_text())
    replay['output'] = settings['output']
    protocol, context, groups, recorded = workload(replay)
    CONTROLLER.clear()
    controller = {'recorded': replay['controller_snapshot'],
                  'template': load_prompt(settings['controller_prompt'])}[settings['controller_source']]
    CONTROLLER.update(controller)
    CONTROLLER['option_template'] = protocol['prompts']['option_template']
    output = Path(settings['output'])
    output.mkdir(parents=True, exist_ok=False)
    save(output / 'protocol.json', {'settings': settings, 'replay': replay, 'controller': CONTROLLER,
                                  'scope': 'Actual recorded finite requests without gold; full vocabulary and unrestricted greedy diagnostic.'})
    shared = SharedConfig.load(replay['shared_config'])
    backend = Backend(shared.backend())
    source = json.loads((Path(replay['source_run']) / replay['runtime_file']).read_text())['backend']
    assert list(backend.answer_labels) == source['answer_labels']
    assert list(backend.answer_label_ids) == source['answer_label_ids']
    generation = GenerationConfig(
        do_sample=settings['do_sample'], max_new_tokens=settings['max_new_tokens'],
        use_cache=settings['use_cache'], eos_token_id=backend.eos_ids,
        pad_token_id=backend.tokenizer.pad_token_id, bos_token_id=backend.tokenizer.bos_token_id)
    reports = []
    for batch_index, group, old in zip(replay['batch_indices'], groups, recorded, strict=True):
        rendered = render(backend, group, context)
        inputs, lengths = backend._encode(rendered['rendered'])
        if settings['verify_recorded_token_counts']:
            assert lengths == [row['input_tokens'] for row in old]
        counts = [len(node['options']) for node in group]
        labels = list(backend.answer_labels[:max(counts)])
        label_ids = list(backend.answer_label_ids[:max(counts)])
        validate_boundaries(backend.tokenizer, rendered['rendered'], rendered['input_ids'], labels, label_ids, counts)
        save(output / f'rendered-{batch_index}.json', rendered)
        mask = inputs['attention_mask']
        hidden_output = backend.model.model(**inputs, position_ids=(mask.cumsum(-1) - 1).clamp_min(0), use_cache=False)
        hidden = hidden_output.last_hidden_state[:, -1].clone()
        del hidden_output
        full = backend.model.lm_head(hidden)
        assert str(full.dtype) == settings['expected_full_head_dtype']
        ids = torch.tensor(label_ids, device=backend.device)
        restricted = F.linear(hidden.float(), backend.model.lm_head.weight.index_select(0, ids).float())
        normalizer = full.float().logsumexp(dim=-1)
        top_logits, top_ids = full.float().topk(settings['top_candidates'], dim=-1)
        rows = []
        for index, (node, count, previous) in enumerate(zip(group, counts, old, strict=True)):
            offered = full[index].index_select(0, ids[:count]).float()
            projected = restricted[index, :count]
            full_choice = int(offered.argmax())
            projected_choice = int(projected.argmax())
            unrestricted_id = int(top_ids[index, 0])
            log_mass = offered.logsumexp(dim=-1) - normalizer[index]
            candidates = token_details(backend.tokenizer, top_ids[index].tolist())
            for candidate, logit in zip(candidates, top_logits[index], strict=True):
                candidate.update(logit=float(logit), probability=float((logit - normalizer[index]).exp()))
            rows.append({'id': node['id'], 'question': node['question'], 'options': node['options'],
                'recorded_result': previous, 'input_tokens': lengths[index],
                'offered_labels': labels[:count], 'offered_token_ids': label_ids[:count],
                'full_head_dtype': str(full.dtype), 'restricted_arithmetic_dtype': str(projected.dtype),
                'full_vocabulary_top_candidates': candidates,
                'full_vocabulary_argmax_is_offered': unrestricted_id in label_ids[:count],
                'offered_total_probability_mass': float(log_mass.exp()), 'offered_log_probability_mass': float(log_mass),
                'offered_full_head_logits': offered.tolist(),
                'offered_full_vocabulary_probabilities': (offered - normalizer[index]).exp().tolist(),
                'full_head_conditional_probabilities': offered.softmax(dim=-1).tolist(),
                'fp32_restricted_logits': projected.tolist(),
                'fp32_restricted_probabilities': projected.softmax(dim=-1).tolist(),
                'full_head_restricted_choice': node['options'][full_choice]['id'],
                'fp32_restricted_choice': node['options'][projected_choice]['id'],
                'restricted_choice_equal': full_choice == projected_choice,
                'maximum_full_head_vs_fp32_restricted_logit_difference': float((offered - projected).abs().max())})
        del hidden, restricted, top_logits, top_ids
        torch.cuda.synchronize(backend.device)
        started = time.perf_counter()
        generated = backend.model.generate(**inputs, generation_config=generation,
            logits_to_keep=CORE['backend']['logits_to_keep'], return_dict_in_generate=True, output_scores=True)
        torch.cuda.synchronize(backend.device)
        elapsed = time.perf_counter() - started
        suffix = generated.sequences[:, inputs['input_ids'].shape[-1]:]
        first_scores = generated.scores[0].float()
        for index, row in enumerate(rows):
            tokens = suffix[index].tolist()
            eos_positions = [position for position, token in enumerate(tokens) if token in backend.eos_ids]
            finished = bool(eos_positions)
            effective = tokens[:eos_positions[0] + 1] if finished else tokens
            row['unconstrained_greedy'] = {
                'token_ids': effective, 'tokens': token_details(backend.tokenizer, effective),
                'text': backend.tokenizer.decode(effective, skip_special_tokens=True),
                'text_with_special_tokens': backend.tokenizer.decode(effective, skip_special_tokens=False),
                'eos_reached': finished, 'truncated': not finished, 'output_tokens': len(effective),
                'first_token_is_offered': effective[0] in row['offered_token_ids'],
                'first_token_equals_full_head_argmax': effective[0] == row['full_vocabulary_top_candidates'][0]['token_id'],
                'maximum_generation_first_scores_vs_full_head_difference': float((first_scores[index] - full[index].float()).abs().max())}
        report = {'batch_index': batch_index, 'batch_size': len(group), 'rows': rows,
                  'generation_seconds': elapsed, 'generation_config': generation.to_dict(),
                  'interpretation': 'Probability mass and arithmetic differences are measurements, without heuristic thresholds or accuracy conclusions.'}
        save(output / f'batch-{batch_index}.json', report)
        reports.append({'batch_index': batch_index, 'batch_size': len(group), 'generation_seconds': elapsed})
        del full, generated, suffix, first_scores, inputs
    completion = {'completed': True, 'batches': reports, 'request_count': sum(map(len, groups)),
                  'source_run': replay['source_run'], 'gold_used': False, 'production_modified': False}
    save(output / 'completion.json', completion)
    print(json.dumps(completion), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--specification', type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.specification.read_text()))
