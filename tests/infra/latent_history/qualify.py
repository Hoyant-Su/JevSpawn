import argparse
import json
from pathlib import Path
import time
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


@torch.inference_mode()
def run(settings):
    shared = SharedConfig.load(settings['shared_config'])
    backend, commands, startup = initialize_parallel(shared, json.loads(Path(settings['parallel_settings']).read_text()))
    method = json.loads(Path(settings['method']).read_text())['settings']
    args = SimpleNamespace(latent_space_realign=True,
        alignment_settings=json.loads(Path(method['alignment_config']).read_text()))
    wrapper = ModelAdapter(backend, args)
    alignment = wrapper._ensure_latent_realign_matrix(wrapper.model, backend.device, args)
    history = RoleZeroHistory(backend, settings['cache'])
    agents = AGENTS.default_agents()
    source = json.loads(Path(settings['source']).read_text())
    output = Path(settings['output'])
    output.mkdir(parents=True, exist_ok=True)
    capacity = shared.model.max_input_tokens + shared.generation.max_new_tokens
    arena = StaticCacheArena(backend.cache_config, shared.runtime.batch_size, capacity,
                            backend.model.lm_head.weight.dtype, backend.device)
    decoders, reports = {}, []

    def compute(workload):
        batch = source[workload['batch']]
        selected = workload['rows']
        rendered = [[wrapper.render_chat(messages) for messages in batch['role_messages'][row]]
                    for row in selected]
        sequences = [backend.tokenizer(texts, add_special_tokens=False, truncation=False)['input_ids']
                     for texts in rendered]
        assert [list(map(len, roles)) for roles in sequences] == [batch['role_input_tokens'][row] for row in selected]
        size = len(selected)
        if size not in decoders:
            decoders[size] = LatentDecode(backend, size, capacity, arena=arena)
        decoder = decoders[size]
        results = {}
        reference = None
        for arm in settings['arms']:
            decoder.cache.reset()
            wrapper.reset()
            constructor = CapturedPaddingTransport if arm == settings['reference_arm'] else CachedRoleTransport
            extras = {} if arm == settings['reference_arm'] else {'history': history,
                'sequences': [roles[settings['role_zero']] for roles in sequences]}
            transport = constructor(backend, decoder, shared.runtime.graph_warmup_steps,
                method['padding_workspace_reserve_bytes'], lambda: None, **extras)
            wrapper.model = transport
            wrapper._latent_realign_matrices = {id(transport): alignment}
            torch.cuda.synchronize(backend.device)
            started = time.perf_counter()
            for role, agent in enumerate(agents):
                ids, mask = padded([roles[role] for roles in sequences], backend.tokenizer.pad_token_id,
                                   backend.device, 'left')
                inputs = {'input_ids': ids, 'attention_mask': mask}
                if agent.role != settings['judger_role']:
                    wrapper.generate_latent_batch(**inputs, latent_steps=method['latent_steps'],
                                                  past_key_values=transport.cache)
                else:
                    transport.phase = settings['judger_phase']
                    transport(**inputs, past_key_values=transport.cache)
            decoder.initialize(transport)
            torch.cuda.synchronize(backend.device)
            elapsed = time.perf_counter() - started
            logits = decoder.logits.clone()
            states = snapshot(decoder, transport.mask.sum(-1).tolist())
            results[arm] = {'elapsed_seconds': elapsed, 'first_output_ids': logits.argmax(-1).tolist(),
                'latent_steps_per_role': method['latent_steps'], 'roles': [agent.role for agent in agents],
                'transport_calls': len(transport.records),
                'history': history.records[-settings['step']] if arm != settings['reference_arm'] else None}
            if arm == settings['reference_arm']:
                reference = (logits, states)
            else:
                results[arm]['logit_max_abs_difference'] = float((logits.float() - reference[settings['zero']].float()).abs().max())
                results[arm]['native_state_max_abs_difference'] = differences(reference[settings['step']], states)
                results[arm]['first_output_equal'] = torch.equal(logits.argmax(-1), reference[settings['zero']].argmax(-1))
        report = {'name': workload['name'], 'source_batch': workload['batch'],
            'task_ids': [batch['task_ids'][row] for row in selected], 'batch_size': size,
            'role_input_tokens': [list(map(len, roles)) for roles in sequences], 'results': results}
        reports.append(report)
        (output / settings['rank_file'].format(rank=dist.get_rank())).write_text(json.dumps(
            {'settings': settings, 'startup': startup, 'reports': reports}, indent=2) + '\n')
        return report

    commands.register(settings['command'], compute)
    if commands.is_leader:
        try:
            for workload in settings['workloads']:
                commands.call(settings['command'], workload)
        finally:
            commands.finish()
    else:
        commands.serve()
    history.history.cache.clear()
    decoders.clear()
    dist.destroy_process_group(commands.control_group)
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    run(json.loads(Path(parser.parse_args().config).read_text()))
