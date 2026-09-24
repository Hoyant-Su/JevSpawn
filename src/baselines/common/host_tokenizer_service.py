import time

from baselines.common.service import BatchService
from jev_spawn.runtime.tokenization import SynchronizedTokenizer


class HostTokenizerBatchService(BatchService):
    def __init__(self, backend, shared, deadlines, *, settings, prompts):
        super().__init__(backend, shared, deadlines, settings=settings, prompts=prompts)
        started = time.perf_counter()
        self.host_tokenizer = SynchronizedTokenizer(backend.tokenizer)
        self.execution_metadata.update(host_tokenizer_initialization_seconds=time.perf_counter() - started,
            host_tokenizer_ownership='One private synchronized host tokenizer; scheduler tokenizer unchanged.')
