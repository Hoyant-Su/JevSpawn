import argparse
import json
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace

import torch
from transformers import StaticCache

from baselines.common.config import SharedConfig
from baselines.common.environment import TaskEnvironment
from baselines.common.tasks import read, rows
from baselines.latentmas.adapter import AGENTS, SOURCE, HybridTransport, ModelAdapter
from baselines.latentmas.native_hybrid_transport import ChunkedHybridTransport
from baselines.latentmas.numerics import difference
from jev_spawn.infra.backend import Backend
from jev_spawn.infra.prompts import load_prompt
from baselines.latentmas.common_service import role_messages
from methods.evidence_flow.environment import EvidenceEnvironment


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


class NativeTransport(HybridTransport):
    def __init__(self, backend, cache):
        super().__init__(backend)
        self.initial_cache = cache

    def __call__(self, input_ids=None, inputs_embeds=None, attention_mask=None,
                 past_key_values=None, **kwargs):
        cache = self.initial_cache if past_key_values is None else past_key_values
        width = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]
        start = int(cache.get_seq_length())
        output = self.trunk(input_ids=input_ids, inputs_embeds=inputs_embeds,
                            attention_mask=attention_mask, past_key_values=cache, **kwargs)
        self.cache, self.mask = output.past_key_values, attention_mask
        self.last_hidden = output.last_hidden_state[:, -1]
        self.records.append({'batch_size': self.last_hidden.shape[0], 'tokens': width,
            'physical_position_start': start, 'physical_position_end': start + width,
            'visible_history_tokens': attention_mask.sum(1).tolist()})
        return output


class MaskPreservingTransport(ChunkedHybridTransport):
    def __init__(self, backend, cache):
        super().__init__(backend)
        self.initial_cache = cache

    def _forward(self, embeddings, mask, cache):
        cache = self.initial_cache if cache is None else cache
        output = super()._forward(embeddings, mask, cache)
        self.records[-1]['visible_history_tokens'] = self.mask.sum(1).tolist()
        self.records[-1]['last_valid_position'] = (self.mask.sum(1) - 1).tolist()
        return output


class ObservedAdapter(ModelAdapter):
    def _apply_latent_realignment(self, hidden, model):
        aligned = super()._apply_latent_realignment(hidden, model)
        self.hidden.append(hidden.detach().clone())
        self.aligned.append(aligned.detach().clone())
        return aligned


def trajectory(backend, batches, settings, alignment, transport_class, deadline):
    capacity = sum(batch['input_ids'].shape[1] for batch in batches) + len(batches) * settings['latent_steps']
    assert capacity <= backend.config['max_input_tokens']
    cache = StaticCache(config=backend.model.config, max_cache_len=capacity)
    transport = transport_class(backend, cache)
    wrapper = ObservedAdapter(backend, SimpleNamespace(latent_space_realign=True))
    wrapper.model = transport
    wrapper._latent_realign_matrices = {id(transport): alignment}
    traces = []
    for index, batch in enumerate(batches):
        assert time.perf_counter() < deadline, 'Declared reference qualification deadline exceeded.'
        wrapper.hidden, wrapper.aligned, transport.records = [], [], []
        started = time.perf_counter()
        wrapper.generate_latent_batch(**batch, latent_steps=settings['latent_steps'],
                                      past_key_values=transport.cache)
        torch.cuda.synchronize(backend.device)
        assert len(wrapper.aligned) == settings['latent_steps']
        assert int(cache.get_seq_length()) == sum(item['input_ids'].shape[1] for item in batches[:index + 1]) + (index + 1) * settings['latent_steps']
        traces.append({'hidden': torch.stack(wrapper.hidden + [transport.last_hidden.clone()], dim=1),
            'aligned': torch.stack(wrapper.aligned, dim=1),
            'logits': backend.model.lm_head(transport.last_hidden), 'seconds': time.perf_counter() - started,
            'forwards': list(transport.records), 'cache_length': int(cache.get_seq_length()),
            'visible_history_tokens': transport.mask.sum(1).tolist()})
    return traces


def compare(left, right):
    return {'hidden': difference(left['hidden'], right['hidden']),
            'aligned': difference(left['aligned'], right['aligned']),
            'logits': difference(left['logits'], right['logits']),
            'left_greedy_ids': left['logits'].argmax(-1).tolist(),
            'right_greedy_ids': right['logits'].argmax(-1).tolist(),
            'greedy_agreement': int((left['logits'].argmax(-1) == right['logits'].argmax(-1)).sum())}


def row(trace, index):
    return {key: trace[key][index:index + 1] for key in ('hidden', 'aligned', 'logits')}


def metadata(traces):
    return [{key: value for key, value in trace.items() if key not in ('hidden', 'aligned', 'logits')}
            for trace in traces]


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config = read(args.config)
    shared = SharedConfig.load(config['shared_config'])
    method, tools = read(config['method']), read(config['environment'])
    tasks = rows(config['tasks'])[config['task_start']:config['task_start'] + config['task_count']]
    assert len(tasks) == shared.runtime.batch_size == config['task_count']
    args.output.mkdir(parents=True, exist_ok=False)
    save(args.output / 'protocol.json', {'config': config, 'method': method,
        'upstream_revision': subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip(),
        'roles': [agent.role for agent in AGENTS.default_agents()[:config['role_count']]],
        'task_ids': [task['task_id'] for task in tasks], 'labels_loaded': False})
    backend = Backend(shared.backend())
    prompts = load_prompt(method['prompts'])
    evidence = EvidenceEnvironment(**tools['evidence'])
    messages = []
    for task in tasks:
        environment = TaskEnvironment(task, tools, args.output / 'tools' / task['task_id'],
                                      deadline=lambda: None, evidence=evidence)
        task_messages = [{'role': 'system', 'content': prompts['system']},
                         {'role': 'user', 'content': environment.reset()}]
        messages.append(role_messages(task_messages, prompts, shared.model.path))
    batches = []
    for role in range(config['role_count']):
        rendered = [backend.tokenizer.apply_chat_template(item[role], tokenize=False,
                    add_generation_prompt=True, enable_thinking=False) for item in messages]
        batches.append(backend.tokenizer(rendered, padding=True, add_special_tokens=False,
                                        return_tensors='pt', truncation=False).to(backend.device))
    save(args.output / 'inputs.json', [{'input_ids': batch['input_ids'].tolist(),
        'attention_mask': batch['attention_mask'].tolist(), 'shape': list(batch['input_ids'].shape)} for batch in batches])
    wrapper = ModelAdapter(backend, SimpleNamespace(latent_space_realign=True))
    alignment = wrapper._ensure_latent_realign_matrix(wrapper.model, backend.device, wrapper.args)
    deadline = time.perf_counter() + config['validation_timeout_seconds']
    native = trajectory(backend, batches, method['settings'], alignment, NativeTransport, deadline)
    save(args.output / 'native_forwards.json', metadata(native))
    corrected = trajectory(backend, batches, method['settings'], alignment, MaskPreservingTransport, deadline)
    save(args.output / 'corrected_forwards.json', metadata(corrected))
    result = {'scope': config['scope'], 'batch_size': len(tasks),
        'batched_comparisons': [compare(left, right) for left, right in zip(corrected, native, strict=True)],
        'rows': [], 'host_tensor_offload': False, 'production_promoted': False}
    for index, task in enumerate(tasks):
        single = [{'input_ids': batch['input_ids'][index][batch['attention_mask'][index].bool()][None],
                   'attention_mask': batch['attention_mask'][index][batch['attention_mask'][index].bool()][None]}
                  for batch in batches]
        independent = trajectory(backend, single, method['settings'], alignment, NativeTransport, deadline)
        result['rows'].append({'task_id': task['task_id'], 'batch_row': index,
            'valid_prompt_lengths': [item['input_ids'].shape[1] for item in single],
            'independent_forwards': metadata(independent),
            'native_b8_vs_unpadded_b1': [compare(row(left, index), right) for left, right in zip(native, independent, strict=True)],
            'corrected_b8_vs_unpadded_b1': [compare(row(left, index), right) for left, right in zip(corrected, independent, strict=True)]})
        save(args.output / 'comparison.json', result)
        print(json.dumps({'task_id': task['task_id'], 'completed_rows': len(result['rows'])}), flush=True)
    assert len(result['rows']) == len(tasks)
    result['completed'] = True
    save(args.output / 'comparison.json', result)


if __name__ == '__main__':
    main()
