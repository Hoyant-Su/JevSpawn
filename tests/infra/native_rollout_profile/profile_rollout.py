import argparse
from itertools import count
import json
from pathlib import Path
import time

import torch.distributed as dist
from torch.profiler import ProfilerActivity, profile, record_function

from baselines.common.parallel_run import execute
from baselines.common.parallel_service import ParallelService
from baselines.common.persistence import save


def run(settings):
    output = Path(settings['profile_output'])
    output.mkdir(parents=True, exist_ok=True)
    indices = count(settings['initial_batch_index'])
    original = ParallelService._parallel_generate

    def observed(service, payload):
        request_type = payload['requests'][0]['type']
        if request_type != settings['request_type']:
            return original(service, payload)
        index = next(indices)
        if index not in settings['profiled_batches']:
            return original(service, payload)
        started = time.perf_counter()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                     **settings['profiler']) as captured:
            dist.barrier(group=service.parallel_commands.control_group)
            with record_function(settings['scope']):
                result = original(service, payload)
        elapsed = time.perf_counter() - started
        rank = dist.get_rank()
        trace = output / settings['trace_file'].format(rank=rank, batch=index)
        captured.export_chrome_trace(str(trace))
        save(output / settings['summary_file'].format(rank=rank, batch=index), {
            'rank': rank, 'batch_index': index,
            'requests': [{key: request['values'][key] for key in settings['recorded_fields']}
                         for request in payload['requests']],
            'elapsed_seconds_with_instrumentation': elapsed, 'trace': str(trace),
            'operators': [{'name': event.key, 'calls': event.count,
                'self_cpu_us': event.self_cpu_time_total,
                'self_device_us': event.self_device_time_total}
                for event in captured.key_averages()],
            'scope': settings['measurement_scope']})
        dist.barrier(group=service.parallel_commands.control_group)
        return result

    ParallelService._parallel_generate = observed
    execute(json.loads(Path(settings['specification']).read_text()),
            Path(settings['run_output']),
            json.loads(Path(settings['parallel_settings']).read_text()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    run(json.loads(parser.parse_args().config.read_text()))
