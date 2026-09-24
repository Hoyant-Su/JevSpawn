from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from queue import Queue
import time
from types import SimpleNamespace

from baselines.common.deadlines import SampleDeadlines
from baselines.latentmas.common_service import LatentMASService


ROOT = Path(__file__).resolve().parents[3]
FIXTURE = {'sample_seconds': 0.4, 'live_admission_delay_seconds': 0.2,
           'delivery_tolerance_seconds': 0.1, 'join_timeout_seconds': 2.0}


def main():
    service = object.__new__(LatentMASService)
    service.shared = SimpleNamespace(generation=SimpleNamespace(max_new_tokens=1, temperature=0.0))
    service.deadlines = SampleDeadlines(FIXTURE['sample_seconds'])
    service.requests = Queue()
    service.timeout_deliveries = {}
    service.forward_deadline_checks = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        service.deadlines.start('expired_fixture')
        expired = pool.submit(service.complete, [], 1, 0.0, task_id='expired_fixture')
        expired_request = service.requests.get(timeout=FIXTURE['join_timeout_seconds'])
        time.sleep(FIXTURE['live_admission_delay_seconds'])
        service.deadlines.start('live_fixture')
        live = pool.submit(service.complete, [], 1, 0.0, task_id='live_fixture')
        live_request = service.requests.get(timeout=FIXTURE['join_timeout_seconds'])
        error = expired.exception(timeout=FIXTURE['join_timeout_seconds'])
        assert isinstance(error, TimeoutError)
        delivered = service.timeout_deliveries['expired_fixture']
        lateness = delivered - service.deadlines.end('expired_fixture')
        assert 0 <= lateness <= FIXTURE['delivery_tolerance_seconds']
        assert not live.done() and not expired_request.future.done()
        service._check_deadlines(['expired_fixture', 'live_fixture'])
        live_request.future.set_result('live request fixture payload')
        assert live.result(timeout=FIXTURE['join_timeout_seconds']) == ['live request fixture payload']
        expired_request.future.set_result('late request fixture payload')
        assert isinstance(expired.exception(), TimeoutError)
    audit = service._deadline_record([expired_request, live_request])
    assert audit['forward_dispatches_retaining_expired_row'] == [1, 0]
    output = ROOT / 'runs/latentmas-deadline-cpu-001'
    output.mkdir(parents=True, exist_ok=False)
    report = {'scope': 'CPU concurrency unit proof of the actual service request wait and deadline bookkeeping. Request payloads and worker delivery are test fixtures; no model inference or GPU-work claim.',
              'fixture': FIXTURE, 'expired_root_released_before_worker_completion': True,
              'live_request_unchanged': True, 'late_worker_result_does_not_replace_timeout': True,
              'timeout_delivery_lateness_seconds': lateness,
              'continued_cohort_dispatch_bookkeeping': audit}
    (output / 'proof.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
