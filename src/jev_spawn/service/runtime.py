from dataclasses import asdict
from functools import partial
import time

import torch

from jev_spawn.service.config import SharedConfig
from jev_spawn.service.deadlines import SampleDeadlines
from jev_spawn.service.parallel_service import parallel_factory
from jev_spawn.service.scheduler import run_tasks
from jev_spawn.service.batch import BatchService
from jev_spawn.infra.backend import Backend


class InferenceRuntime:
    def __init__(self, config_path, service_factory=None, backend=None):
        self.config = SharedConfig.load(config_path)
        factory = BatchService if service_factory is None else service_factory
        if self.config.runtime.world_size > 1:
            factory = parallel_factory(factory)
        started = time.perf_counter()
        self.backend_reused = backend is not None
        if backend is None:
            self.backend = Backend(self.config.backend())
            torch.cuda.synchronize(self.backend.device)
            self.load_seconds = time.perf_counter() - started
        else:
            assert backend.config == self.config.backend(), 'Injected backend configuration differs.'
            self.backend = backend
            self.load_seconds = 0.0
        self.deadlines = SampleDeadlines(self.config.runtime.sample_timeout_seconds)
        started = time.perf_counter()
        self.service = factory(self.backend, self.config, self.deadlines)
        self.service_load_seconds = time.perf_counter() - started

    def complete(self, task_id):
        return partial(self.service.complete, task_id=task_id)

    def run(self, tasks, solve, commit):
        assert len({task['task_id'] for task in tasks}) == len(tasks)
        return run_tasks(tasks, solve, self.config.runtime.root_batch_size, self.deadlines, commit)

    def metadata(self):
        return {'shared_config': asdict(self.config), 'backend': self.backend.metadata,
                'model_load_seconds': self.load_seconds, 'backend_reused': self.backend_reused,
                'service_load_seconds': self.service_load_seconds,
                'service': self.service.execution_metadata,
                'root_scheduling': 'bounded rolling admission',
                'generation_scheduling': self.config.runtime.generation_scheduling,
                'continuous_decode_batching': self.config.runtime.generation_scheduling == 'continuous'}

    def close(self):
        self.service.close()
