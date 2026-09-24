import argparse
from functools import partial
import json
from pathlib import Path
from unittest.mock import patch

import torch
import torch.distributed as dist
import yaml

from baselines.common.config import SharedConfig
from baselines.common.parallel_run import initialize_parallel
from baselines.common.runtime import InferenceRuntime
from baselines.common.service import BatchService
from jev_spawn.runtime.refill_state import RefillState
from qualify_service import measure, save


PREFILL = RefillState.prefill
COMPACT = RefillState.compact
CHECKS = []


def state_tensors(state):
    return [*state.arena.row_tensors(), *state.row_tensors()]


def checked_prefill(state, *args, **kwargs):
    live = state.arena.live_count
    expected = [tensor[:live].clone() for tensor in state_tensors(state)]
    result = PREFILL(state, *args, **kwargs)
    assert all(torch.equal(before, after[:live]) for before, after in zip(expected, state_tensors(state), strict=True))
    CHECKS.append({'operation': 'prefill_preserves_survivors', 'rows': live})
    return result


def checked_compact(state, survivors):
    index = torch.tensor(survivors, device=state.arena.device, dtype=torch.long)
    expected = [tensor.index_select(0, index) for tensor in state_tensors(state)]
    requests = [state.requests[row] for row in survivors]
    COMPACT(state, survivors)
    assert state.requests == requests
    assert all(torch.equal(before, after[:len(survivors)]) for before, after in zip(expected, state_tensors(state), strict=True))
    CHECKS.append({'operation': 'compact_preserves_full_state', 'survivors': survivors})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--specification', type=Path, required=True)
    args = parser.parse_args()
    settings = json.loads(args.specification.read_text())
    output = Path(settings['output'])
    reference = yaml.safe_load(Path(settings['shared_config']).read_text())
    candidate = yaml.safe_load(Path(settings['candidate_config']).read_text())
    expected = yaml.safe_load(Path(settings['shared_config']).read_text())
    expected['runtime'].update(generation_scheduling='continuous', cache_allocation='shared_refill_cache_arena_v1')
    assert expected == candidate
    shared = SharedConfig.load(settings['shared_config'])
    backend, commands, starts = initialize_parallel(shared, json.loads(Path(settings['parallel_settings']).read_text()))
    if commands.is_leader:
        output.mkdir(parents=True, exist_ok=False)
        save(output / 'specification.json', settings)
        save(output / 'parallel_startup.json', {'ranks': starts})
    results = {}
    for arm_specification in settings['arms']:
        arm, config = arm_specification['name'], arm_specification['config']
        prefill = checked_prefill if settings['verify_membership_state'] else PREFILL
        compact = checked_compact if settings['verify_membership_state'] else COMPACT
        with patch.object(RefillState, 'prefill', prefill), patch.object(RefillState, 'compact', compact):
            if commands.is_leader:
                try:
                    results[arm] = measure(settings, config, backend, output / arm)
                finally:
                    commands.finish()
            else:
                runtime = InferenceRuntime(config, partial(BatchService, settings={}, prompts={}), backend=backend)
                try:
                    commands.serve()
                finally:
                    runtime.close()
        dist.barrier(group=commands.control_group)
        commands.handlers.clear()
    rank_checks = [None] * shared.runtime.world_size
    dist.all_gather_object(rank_checks, CHECKS, group=commands.control_group)
    if commands.is_leader:
        before, after = results['cohort'], results['continuous']
        agreement = {identity: before['rows'][identity]['token_ids'] == row['token_ids']
                     and before['rows'][identity]['finish_reason'] == row['finish_reason']
                     for identity, row in after['rows'].items()}
        summary = {'exact_output_and_stop_agreement': agreement,
            'all_outputs_and_stops_equal': all(agreement.values()),
            'refill_before_survivor_finished': after['admitted_before_survivor_finished'],
            'cohort_seconds': before['elapsed_seconds'], 'continuous_seconds': after['elapsed_seconds'],
            'cohort_peak_allocated_bytes': before['peak_allocated_bytes'],
            'continuous_peak_allocated_bytes': after['peak_allocated_bytes'],
            'capacity': reference['model']['max_input_tokens'] + reference['generation']['max_new_tokens'],
            'membership_state_checks_by_rank': rank_checks,
            'scope': 'Real TP4 optimized request replay; explicit64/256-token profiling budgets. State-check overhead is included when enabled; not full-task accuracy.'}
        save(output / 'completion.json', summary)
        print(json.dumps(summary), flush=True)
    dist.destroy_process_group(commands.control_group)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
