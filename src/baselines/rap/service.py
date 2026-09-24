"""Serialize RAP reward readouts with generation on the shared model replica."""

from concurrent.futures import Future
from dataclasses import dataclass
import time

import torch

from baselines.official.model_service import GenerationService


@dataclass
class ProbabilityRequest:
    messages: list
    labels: tuple
    task_id: str
    submitted: float
    future: Future

    @property
    def signature(self):
        return 'next_token_probabilities', self.labels


class RAPService(GenerationService):
    def next_token_probabilities(self, messages, labels, *, task_id):
        requests = [ProbabilityRequest(row, tuple(labels), task_id, time.perf_counter(), Future())
                    for row in messages]
        for request in requests:
            self.requests.put(request)
        return [request.future.result() for request in requests]

    @torch.inference_mode()
    def _generate(self, batch):
        if not isinstance(batch[0], ProbabilityRequest):
            return super()._generate(batch)
        backend = self.backend
        torch.cuda.synchronize(backend.device)
        torch.cuda.reset_peak_memory_stats(backend.device)
        started = time.perf_counter()
        rendered = backend.tokenizer.apply_chat_template(
            [r.messages for r in batch], tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
        inputs, input_tokens = backend._encode(rendered)
        label_ids = backend.tokenizer(list(batch[0].labels), add_special_tokens=False)['input_ids']
        assert all(len(ids) == 1 for ids in label_ids)
        logits = backend.model(**inputs, use_cache=False, logits_to_keep=1).logits[:, -1]
        values = logits[:, [ids[0] for ids in label_ids]].float().softmax(-1).cpu().tolist()
        del logits
        torch.cuda.synchronize(backend.device)
        self.records.append({
            'operation': 'next_token_probabilities', 'labels': list(batch[0].labels),
            'task_ids': [r.task_id for r in batch], 'batch_size': len(batch),
            'messages': [r.messages for r in batch], 'probabilities': values,
            'elapsed_seconds': time.perf_counter() - started,
            'queue_seconds': [started - r.submitted for r in batch],
            'input_tokens': input_tokens, 'output_tokens': [0] * len(batch),
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(backend.device),
            'peak_reserved_bytes': torch.cuda.max_memory_reserved(backend.device), 'decode': [],
        })
        return values
