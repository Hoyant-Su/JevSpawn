import argparse
from itertools import count
import json
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

from tests.infra.history_cache.replay_checkpoint import ObservedCheckpoint, ObservedReference
from tests.infra.history_cache.replay_finite import reset_history
from tests.infra.native_history_cache import replay


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    settings = json.loads(parser.parse_args().config.read_text())
    original_measure = replay.measure
    indices = count()

    def measured(operation, device, repetitions):
        index = next(indices)
        rank = torch.distributed.get_rank()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as trace:
            result = original_measure(operation, device, repetitions)
        target = Path(settings['output']) / settings['trace_file'].format(rank=rank, index=index)
        trace.export_chrome_trace(str(target))
        rows = [{'name': event.key, 'calls': event.count,
                 'self_cpu_us': event.self_cpu_time_total,
                 'self_cuda_us': event.self_device_time_total,
                 'input_shapes': event.input_shapes}
                for event in trace.key_averages(group_by_input_shape=True)]
        target.with_suffix('.summary.json').write_text(json.dumps(rows, indent=2) + '\n')
        return result

    replay.HistoryTail = ObservedCheckpoint
    replay.StableFiniteGraphTail = ObservedReference
    replay.reset_history = reset_history
    replay.measure = measured
    replay.run(settings)
