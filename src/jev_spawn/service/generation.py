"""Serve upstream agent model calls through one shared batched model replica."""

from concurrent.futures import Future
from copy import deepcopy
from dataclasses import dataclass, field
from queue import Empty, Queue
from threading import Thread
import time

import torch
from transformers import GenerationConfig, StopStringCriteria, StoppingCriteria

from jev_spawn.service.errors import InputLimitError


class RowStops(StoppingCriteria):
    def __init__(self, tokenizer, stops, eos_ids, prompt_width, batch_size, device):
        self.strings = StopStringCriteria(tokenizer, list(stops)) if stops else None
        self.eos = torch.tensor(eos_ids, device=device)
        self.prompt_width = prompt_width
        self.lengths = torch.zeros(batch_size, dtype=torch.long, device=device)

    def __call__(self, input_ids, scores, **kwargs):
        done = torch.isin(input_ids[:, -1], self.eos)
        if self.strings is not None:
            done |= self.strings(input_ids[:, self.prompt_width:], scores)
        self.lengths = torch.where(done & (self.lengths == 0),
                                   input_ids.shape[1] - self.prompt_width, self.lengths)
        return done


@dataclass
class Request:
    messages: list
    max_tokens: int
    temperature: float
    stop: tuple
    task_id: str
    submitted: float
    future: Future
    input_ids: list = field(default_factory=list, init=False)

    @property
    def signature(self):
        return self.max_tokens, self.temperature, self.stop


class GenerationService:
    def __init__(self, backend, batch_size, batch_wait_seconds):
        self.backend = backend
        self.batch_size = batch_size
        self.batch_wait_seconds = batch_wait_seconds
        self.requests = Queue()
        self.records = []
        self.pending = []
        self.input_failures = []
        self.thread = Thread(target=self._serve)
        self.thread.start()

    def complete(self, messages, max_tokens, temperature, n=1, stop=None, *, task_id):
        stops = (stop,) if isinstance(stop, str) else tuple(stop or ())
        requests = [Request(deepcopy(messages), max_tokens, temperature, stops, task_id,
                            time.perf_counter(), Future()) for _ in range(n)]
        for request in requests:
            self.requests.put(request)
        return [request.future.result() for request in requests]

    def close(self):
        self.requests.put(None)
        self.thread.join()

    def _serve(self):
        while True:
            first = self.pending.pop(0) if self.pending else self.requests.get()
            if first is None:
                return
            batch = [first]
            compatible = [r for r in self.pending if r.signature == first.signature]
            selected = compatible[:self.batch_size - 1]
            batch.extend(selected)
            self.pending = [r for r in self.pending if all(r is not chosen for chosen in selected)]
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
                if request.signature == first.signature:
                    batch.append(request)
                else:
                    self.pending.append(request)
            validation_started = time.perf_counter()
            validation_batch_size = len(batch)
            try:
                batch = self._validate_inputs(batch)
            except Exception as error:
                for request in batch:
                    if not request.future.done():
                        request.future.set_exception(error)
                continue
            validation_seconds = time.perf_counter() - validation_started
            if not batch:
                continue
            record_start = len(self.records)
            try:
                texts = self._generate(batch)
            except Exception as error:
                for request in batch:
                    if not request.future.done():
                        request.future.set_exception(error)
            else:
                for record in self.records[record_start:]:
                    record.setdefault('input_validation_seconds', validation_seconds)
                    record.setdefault('input_validation_batch_size', validation_batch_size)
                for request, text in zip(batch, texts):
                    if not request.future.done():
                        request.future.set_result(text)

    def _record_batch(self, batch, record):
        for request in batch:
            request.record = record
        self.records.append(record)

    def _validate_inputs(self, batch):
        tokenizer = self.backend.tokenizer
        rendered = tokenizer.apply_chat_template(
            [r.messages for r in batch], tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
        encoded = tokenizer(rendered, padding=False, truncation=False,
                            add_special_tokens=False)
        limit = self.backend.config['max_input_tokens']
        valid = []
        for request, text, tokens in zip(batch, rendered, encoded['input_ids'], strict=True):
            length = len(tokens)
            if length > limit:
                error = InputLimitError(
                    f'Input has {length} tokens; limit is {limit}. Inputs are never truncated.')
                self.input_failures.append({
                    'task_id': request.task_id, 'messages': request.messages,
                    'input_tokens': length, 'max_input_tokens': limit,
                    'error': str(error),
                })
                request.future.set_exception(error)
            else:
                self._admit_input(request, text, tokens)
                valid.append(request)
        return valid

    def _admit_input(self, request, rendered, tokens):
        request.input_ids = tokens

    def _stopping(self, batch, prompt_width):
        return RowStops(self.backend.tokenizer, batch[0].stop, self.backend.eos_ids,
                        prompt_width, len(batch), self.backend.device)

    def _generate_tokens(self, inputs, options, stopping):
        events = []

        def record_token(module, inputs, output):
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            events.append(event)

        hook = self.backend.model.register_forward_hook(record_token)
        try:
            sequences = self.backend.model.generate(
                **inputs, generation_config=GenerationConfig(**options),
                stopping_criteria=[stopping], logits_to_keep=1)
        finally:
            hook.remove()
        return sequences, events

    @torch.inference_mode()
    def _generate(self, batch):
        backend = self.backend
        torch.cuda.synchronize(backend.device)
        torch.cuda.reset_peak_memory_stats(backend.device)
        started = time.perf_counter()
        input_tokens = [len(request.input_ids) for request in batch]
        assert min(input_tokens) > 0
        inputs = backend.tokenizer.pad(
            {'input_ids': [request.input_ids for request in batch],
             'attention_mask': [[1] * length for length in input_tokens]},
            padding=True, return_tensors='pt').to(backend.device)
        first = batch[0]
        options = {
            'do_sample': first.temperature > 0,
            'max_new_tokens': max(request.max_tokens for request in batch),
            'use_cache': True,
            'eos_token_id': backend.eos_ids,
            'pad_token_id': backend.tokenizer.pad_token_id,
            'bos_token_id': backend.tokenizer.bos_token_id,
        }
        if first.temperature > 0:
            options.update(temperature=first.temperature, top_k=0, top_p=1.0)
        stopping = self._stopping(batch, inputs['input_ids'].shape[1])
        origin = torch.cuda.Event(enable_timing=True)
        origin.record()
        compute_started = time.perf_counter()

        sequences, events = self._generate_tokens(inputs, options, stopping)
        generated = sequences[:, inputs['input_ids'].shape[1]:]
        finished = stopping.lengths.tolist()
        counts = [count or generated.shape[1] for count in finished]
        truncated = [count == 0 for count in finished]
        token_ids = generated.tolist()
        token_ids = [row[:count] for row, count in zip(token_ids, counts)]
        texts = backend.tokenizer.batch_decode(token_ids, skip_special_tokens=True)
        for index in range(len(texts)):
            positions = [texts[index].find(stop) for stop in batch[index].stop if stop in texts[index]]
            if positions:
                texts[index] = texts[index][:min(positions)]
        torch.cuda.synchronize(backend.device)
        elapsed = time.perf_counter() - started
        times = [compute_started + origin.elapsed_time(event) / 1000 for event in events]
        self._record_batch(batch, {
            'task_ids': [r.task_id for r in batch], 'batch_size': len(batch),
            'messages': [r.messages for r in batch],
            'max_new_tokens': options['max_new_tokens'], 'temperature': first.temperature,
            'requested_max_new_tokens': [request.max_tokens for request in batch],
            'stop': list(first.stop), 'elapsed_seconds': elapsed,
            'queue_seconds': [started - r.submitted for r in batch],
            'input_tokens': input_tokens, 'output_tokens': counts, 'truncated': truncated,
            'output_token_ids': token_ids,
            'texts': texts,
            'started_monotonic': started,
            'finished_monotonic': started + elapsed,
            'preparation_seconds': compute_started - started,
            'forward_timing': 'CUDA events; completion metadata is synchronized for stopping',
            'decode_steps': len(events),
            'active_token_slots': sum(counts),
            'executed_token_slots': len(batch) * generated.shape[1],
            'row_tail_wait_seconds': [started + elapsed - times[count - 1] for count in counts],
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(backend.device),
            'peak_reserved_bytes': torch.cuda.max_memory_reserved(backend.device),
            'decode': [{'ttft_seconds': times[0] - started,
                        'inter_token_seconds': [b - a for a, b in zip(times[:count], times[1:count])]}
                       for count in counts],
        })
        return texts
