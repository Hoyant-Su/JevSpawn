import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from queue import Queue
import time
from types import SimpleNamespace

from baselines.common.deadlines import SampleDeadlines
from baselines.common.service import BatchService


def check(config):
    service = object.__new__(BatchService)
    service.shared = SimpleNamespace(generation=SimpleNamespace(
        max_new_tokens=config['max_new_tokens'], temperature=config['temperature']))
    service.deadlines = SampleDeadlines(config['sample_seconds'])
    service.requests = Queue()
    service.timeout_deliveries = {}

    async def agent_request(identity):
        return await asyncio.to_thread(service.complete, [], config['max_new_tokens'],
            config['temperature'], n=config['completion_samples'], task_id=identity)

    with ThreadPoolExecutor(max_workers=config['root_workers']) as pool:
        expired_id, live_id = config['timeout_task_id'], config['live_task_id']
        service.deadlines.start(expired_id)
        expired = pool.submit(asyncio.run, agent_request(expired_id))
        expired_requests = [service.requests.get(timeout=config['join_timeout_seconds'])
                            for _ in range(config['completion_samples'])]
        time.sleep(config['live_admission_delay_seconds'])
        service.deadlines.start(live_id)
        live = pool.submit(service.complete, [], config['max_new_tokens'], config['temperature'],
                           n=config['completion_samples'], task_id=live_id)
        live_requests = [service.requests.get(timeout=config['join_timeout_seconds'])
                         for _ in range(config['completion_samples'])]
        error = expired.exception(timeout=config['join_timeout_seconds'])
        observed = time.perf_counter()
        assert isinstance(error, TimeoutError) and expired_id in str(error)
        lateness = observed - service.deadlines.end(expired_id)
        assert 0 <= lateness <= config['delivery_tolerance_seconds']
        assert not live.done() and all(not request.future.done() for request in expired_requests)
        for request in live_requests:
            request.future.set_result(config['live_payload'])
        assert live.result(timeout=config['join_timeout_seconds']) == [config['live_payload']] * config['completion_samples']
        for request in expired_requests:
            request.future.set_result(config['late_payload'])
        assert isinstance(expired.exception(), TimeoutError)
    return {'asyncio_root_released_before_worker_completion': True,
            'live_multi_sample_request_unchanged': True, 'no_worker_futures_cancelled': True,
            'late_worker_result_does_not_replace_timeout': True,
            'observed_timeout_lateness_seconds': lateness}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    result = {'scope': 'CPU request concurrency regression using actual BatchService.complete and AgentPrune asyncio-to-thread pattern; no model or GPU-work claim.',
              'config': config, **check(config)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
