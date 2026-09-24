import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist

from baselines.common.config import SharedConfig
from baselines.common.parallel_run import initialize_parallel
from baselines.latentmas.adapter import AGENTS, ModelAdapter
from baselines.latentmas.common_service import LatentDecode
from baselines.latentmas.native_hybrid_transport import CapturedPaddingTransport
from jev_spawn.algo.structured import padded
from jev_spawn.runtime.cache_arena import StaticCacheArena
from tests.infra.history_cache.qualify import differences, snapshot
from tests.infra.latent_history.candidate import CachedRoleTransport, RoleZeroHistory


class Observer:
    def __init__(self, settings, history, sequences):
        self.settings, self.history, self.sequences = settings, history, sequences
        self.references, self.states, self.reports = {}, None, {}

    def begin(self, arm):
        self.arm, self.calls, self.alignments = arm, [], []
        self.reference = arm == self.settings['reference_arm']

    def compare(self, name, tensor):
        key = (name, len(self.calls), len(self.alignments))
        if self.reference:
            self.references[key] = tensor.clone()
        original = self.references[key]
        return {'shape': list(tensor.shape), 'stride': list(tensor.stride()),
                'max_abs_difference': float((original.float() - tensor.float()).abs().max()),
                'row_max_abs_difference': (original.float() - tensor.float()).abs().flatten(1).amax(-1).tolist(),
                'equal': torch.equal(original, tensor)}

    def observed(self, transport, output):
        terminal = torch.stack([layer[:, self.settings['last_index']] for layer in output.hidden_states],
                               dim=self.settings['layer_axis'])
        record = {'phase': transport.phase, 'terminal': self.compare('terminal', terminal),
                  'last_hidden': self.compare('last_hidden', output.hidden_states[self.settings['last_index']][:, self.settings['last_index']]),
                  'hidden_tensor_strides': [list(layer.stride()) for layer in output.hidden_states]}
        if not self.calls:
            record['native_row_to_first_terminal_difference'] = (terminal.float() - terminal[:self.settings['step']].float()).abs().flatten(1).amax(-1).tolist()
            states = snapshot(transport.decoder, list(map(len, self.sequences)))
            if self.reference:
                self.states = states
                self.history.history.save_rows(SimpleNamespace(cache=transport.cache, logits=terminal),
                                               self.sequences, list(range(len(self.sequences))))
                saved = self.history.history.cache.outputs[tuple(self.sequences[self.settings['zero']])]
                record['saved_first_row_terminal_difference'] = float((saved.float() - terminal[:self.settings['step']].float()).abs().max())
            record['state_difference'] = differences(self.states, states)
            per_row = {name: [] for name, _ in states}
            for (name, original), (_, restored) in zip(self.states, states, strict=True):
                delta = (original.float() - restored.float()).abs()
                per_row[name].append(delta.flatten(1).amax(-1))
            record['state_row_differences'] = {
                name: (torch.cat(values).view(-1, len(self.sequences)).amax(0).tolist()
                       if name == self.settings['kv_name'] else torch.stack(values).amax(0).tolist())
                for name, values in per_row.items()}
        self.calls.append(record)

    def realignment(self, hidden, output):
        self.alignments.append({'input': self.compare('alignment_input', hidden),
                                'output': self.compare('alignment_output', output)})

    def finish(self, logits):
        self.reports[self.arm] = {'forward_calls': self.calls, 'realignments': self.alignments,
                                 'judger_logits': self.compare('judger_logits', logits)}


class ObservedTransport:
    def __call__(self, *args, **kwargs):
        output = super().__call__(*args, **kwargs)
        self.observer.observed(self, output)
        return output


class NativeTransport(ObservedTransport, CapturedPaddingTransport):
    pass


class RestoredTransport(ObservedTransport, CachedRoleTransport):
    pass


class ObservedModel(ModelAdapter):
    def _apply_latent_realignment(self, hidden, model):
        output = super()._apply_latent_realignment(hidden, model)
        self.observer.realignment(hidden, output)
        return output


@torch.inference_mode()
def run(settings):
    shared = SharedConfig.load(settings['shared_config'])
    backend, commands, startup = initialize_parallel(shared, json.loads(Path(settings['parallel_settings']).read_text()))
    method = json.loads(Path(settings['method']).read_text())['settings']
    args = SimpleNamespace(latent_space_realign=True,
        alignment_settings=json.loads(Path(method['alignment_config']).read_text()))
    wrapper = ObservedModel(backend, args)
    alignment = wrapper._ensure_latent_realign_matrix(wrapper.model, backend.device, args)
    history = RoleZeroHistory(backend, settings['cache'])
    source = json.loads(Path(settings['source']).read_text())[settings['workload']['batch']]
    rows = settings['workload']['rows']
    rendered = [[wrapper.render_chat(messages) for messages in source['role_messages'][row]] for row in rows]
    sequences = [backend.tokenizer(texts, add_special_tokens=False, truncation=False)['input_ids'] for texts in rendered]
    assert [list(map(len, roles)) for roles in sequences] == [source['role_input_tokens'][row] for row in rows]
    role_zero = [roles[settings['role_zero']] for roles in sequences]
    observer = Observer(settings, history, role_zero)
    wrapper.observer = observer
    capacity = shared.model.max_input_tokens + shared.generation.max_new_tokens
    arena = StaticCacheArena(backend.cache_config, shared.runtime.batch_size, capacity,
                            backend.model.lm_head.weight.dtype, backend.device)
    decoder = LatentDecode(backend, len(rows), capacity, arena=arena)
    output = Path(settings['output'])
    output.mkdir(parents=True, exist_ok=True)

    def compute(_):
        for arm in settings['arms']:
            decoder.cache.reset()
            wrapper.reset()
            observer.begin(arm)
            constructor = RestoredTransport if arm == settings['restored_arm'] else NativeTransport
            extras = {'history': history, 'sequences': role_zero} if arm == settings['restored_arm'] else {}
            transport = constructor(backend, decoder, shared.runtime.graph_warmup_steps,
                method['padding_workspace_reserve_bytes'], lambda: None, **extras)
            transport.observer = observer
            wrapper.model = transport
            wrapper._latent_realign_matrices = {id(transport): alignment}
            for role, agent in enumerate(AGENTS.default_agents()):
                ids, mask = padded([roles[role] for roles in sequences], backend.tokenizer.pad_token_id, backend.device, 'left')
                inputs = {'input_ids': ids, 'attention_mask': mask}
                if agent.role != settings['judger_role']:
                    wrapper.generate_latent_batch(**inputs, latent_steps=method['latent_steps'], past_key_values=transport.cache)
                else:
                    transport.phase = settings['judger_phase']
                    transport(**inputs, past_key_values=transport.cache)
            decoder.initialize(transport)
            observer.finish(decoder.logits)
            (output / settings['rank_file'].format(rank=dist.get_rank())).write_text(json.dumps(
                {'settings': settings, 'startup': startup, 'task_ids': [source['task_ids'][row] for row in rows],
                 'reports': observer.reports}, indent=2) + '\n')
        return observer.reports

    commands.register(settings['command'], compute)
    if commands.is_leader:
        try:
            commands.call(settings['command'], {})
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
