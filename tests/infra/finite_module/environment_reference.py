import argparse
import json
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F

from baselines.common.config import SharedConfig
from baselines.common.parallel_run import initialize_parallel
from jev_spawn.algo.structured import padded
from jev_spawn.infra import backend as backend_module
from jev_spawn.infra.prompts import load_prompt
from jev_spawn.schema import CONTROLLER, controller_prompts


def read(path):
    return json.loads(Path(path).read_text())


def retain_native_execution(model, settings):
    return model


@torch.inference_mode()
def run(config):
    shared = SharedConfig.load(config['shared_config'])
    installers = {'optimized': backend_module.install_qwen35_execution, 'native': retain_native_execution}
    backend_module.install_qwen35_execution = installers[config['model_operators']]
    backend, commands, startup = initialize_parallel(shared, read(config['parallel_settings']))
    method = read(config['method'])
    CONTROLLER['option_template'] = load_prompt(method['prompts'])['option_template']
    records = []
    for selection in config['selections']:
        source = read(Path(config['source']) / config['task_file'].format(index=selection['task']))
        transition = source['transitions'][selection['turn']]
        fields = transition['checks']
        sequences = []
        for field in fields:
            state = CONTROLLER['state_template'].format(context=field['context'], state=field['state'])
            user, = controller_prompts([state], field['question'], field['options'],
                backend.answer_labels[:len(field['options'])], CONTROLLER['output_instruction'])
            messages = [{'role': 'system', 'content': CONTROLLER['system']}, {'role': 'user', 'content': user}]
            rendered = backend.tokenizer.apply_chat_template(messages, tokenize=False,
                add_generation_prompt=True, enable_thinking=shared.generation.enable_thinking)
            sequences.append(backend.tokenizer(rendered, add_special_tokens=False)['input_ids'])
        assert list(map(len, sequences)) == [row['input_tokens'] for row in transition['readings']]
        ids, mask = padded(sequences, backend.tokenizer.pad_token_id, backend.device, config['padding'])
        output = backend.model.model(input_ids=ids, attention_mask=mask,
            position_ids=(mask.cumsum(-1) - 1).clamp_min(0), use_cache=False)
        count = max(len(field['options']) for field in fields)
        logits = F.linear(output.last_hidden_state[:, -1].float(), backend.finite_output_weights[:count])
        vocabulary = backend.model.lm_head(output.last_hidden_state[:, -1]).float()
        distribution = vocabulary.softmax(-1)
        label_ids = torch.tensor(backend.answer_label_ids[:count], device=backend.device)
        mass = distribution.index_select(-1, label_ids).sum(-1).tolist()
        top_probabilities, top_ids = distribution.topk(config['top_tokens'], dim=-1)
        top_text = [[backend.tokenizer.decode([token]) for token in row] for row in top_ids.tolist()]
        rows = logits.tolist()
        comparisons = [{'id': field['id'], 'previous': previous['choice'],
            'reference': field['options'][max(range(len(field['options'])), key=lambda index: row[index])]['id'],
            'previous_logits': previous['option_logits'], 'reference_logits': row,
            'max_logit_difference': max(abs(a-b) for a,b in zip(previous['option_logits'], row, strict=True))}
            for field, previous, row in zip(fields, transition['readings'], rows, strict=True)]
        for comparison, probability, tokens, probabilities in zip(
                comparisons, mass, top_text, top_probabilities.tolist(), strict=True):
            comparison.update(label_probability_mass=probability, top_tokens=tokens,
                              top_probabilities=probabilities)
        records.append({'selection': selection, 'comparisons': comparisons})
        if commands.is_leader:
            destination = Path(config['output'])
            destination.mkdir(parents=True, exist_ok=True)
            (destination / config['result_file']).write_text(json.dumps(
                {'config': config, 'startup': startup, 'records': records}, indent=2)+'\n')
            print(json.dumps(records[-1]), flush=True)
    dist.destroy_process_group(commands.control_group)
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    run(read(parser.parse_args().config))
