import json
import time

import pynvml


def process_memory(pid):
    values = []
    for index in range(pynvml.nvmlDeviceGetCount()):
        handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        for process in pynvml.nvmlDeviceGetComputeRunningProcesses(handle):
            if process.pid == pid:
                assert isinstance(process.usedGpuMemory, int)
                assert 0 <= process.usedGpuMemory < pynvml.NVML_VALUE_NOT_AVAILABLE_ulonglong.value
                values.append({'gpu_uuid': pynvml.nvmlDeviceGetUUID(handle),
                               'physical_bytes': process.usedGpuMemory})
    assert len(values) <= 1, 'The trial must use one physical GPU.'
    return values[0] if values else None


class ProcessMemory:
    def __init__(self, output):
        pynvml.nvmlInit()
        self.path = output
        self.stream = None
        self.samples = 0
        self.peak_bytes = None
        self.pid = None

    def start(self, pid):
        self.pid, self.origin = pid, time.perf_counter()
        self.stream = self.path.open('x')

    def sample(self):
        value = process_memory(self.pid)
        if value is not None:
            value.update(pid=self.pid, seconds=time.perf_counter() - self.origin)
            self.samples += 1
            self.peak_bytes = value['physical_bytes'] if self.peak_bytes is None else max(self.peak_bytes, value['physical_bytes'])
            self.stream.write(json.dumps(value) + '\n')
            self.stream.flush()

    def close(self):
        self.stream.close()
        pynvml.nvmlShutdown()
        return {'samples': self.samples, 'sampled_peak_physical_bytes': self.peak_bytes,
                'scope': 'NVML process memory sampled by the parent during startup and inference. The sampled peak is not an exact continuous maximum.'}
