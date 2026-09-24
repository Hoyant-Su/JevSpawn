from concurrent.futures import Future
from queue import Empty, Queue
from threading import Thread
import time

import torch

from baselines.official.model_service import GenerationService


class SharedGenerationService(GenerationService):
    def __init__(self, backend, batch_size, batch_wait_seconds, gpu_lock):
        self.gpu_lock = gpu_lock
        super().__init__(backend, batch_size, batch_wait_seconds)

    def _generate(self, batch):
        with self.gpu_lock:
            return super()._generate(batch)


class EmbeddingService:
    def __init__(self, index, batch_size, batch_wait_seconds, gpu_lock):
        self.index, self.batch_size = index, batch_size
        self.batch_wait_seconds, self.gpu_lock = batch_wait_seconds, gpu_lock
        self.requests, self.records = Queue(), []
        self.thread = Thread(target=self._serve)
        self.thread.start()

    def complete(self, texts, task_id):
        assert isinstance(texts, list) and len(texts) == 1
        future = Future()
        self.requests.put((texts[0], task_id, future))
        return future.result()

    def close(self):
        self.requests.put(None)
        self.thread.join()

    def _serve(self):
        while True:
            first = self.requests.get()
            if first is None:
                return
            batch = [first]
            deadline = time.perf_counter() + self.batch_wait_seconds
            while len(batch) < self.batch_size:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                try:
                    request = self.requests.get(timeout=remaining)
                except Empty:
                    break
                if request is None:
                    self.requests.put(None)
                    break
                batch.append(request)
            try:
                with self.gpu_lock:
                    torch.cuda.synchronize(self.index.device)
                    started = time.perf_counter()
                    texts = [row[0] for row in batch]
                    vectors = self.index.encode(texts).tolist()
                    counts = [len(row) for row in self.index.tokenizer(texts)['input_ids']]
                    torch.cuda.synchronize(self.index.device)
                    self.records.append({'task_ids': [row[1] for row in batch], 'batch_size': len(batch),
                        'elapsed_seconds': time.perf_counter() - started, 'input_tokens': counts,
                        'queries': texts})
            except Exception as error:
                for _, _, future in batch:
                    future.set_exception(error)
            else:
                for (_, _, future), vector, count in zip(batch, vectors, counts):
                    future.set_result({'vectors': [vector], 'input_tokens': count})
