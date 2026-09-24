from functools import partial
from queue import Empty
import time

import torch

from baselines.common.rolling_service import Entry, RollingService, Wave
from baselines.common.service import BatchService
from jev_spawn.runtime.refill_cache_arena import RefillCacheArena
from jev_spawn.runtime.refill_state import RefillState


class RefillWave(Wave):
    def __init__(self, service, decoder, entries, start, stop):
        state = service.refill
        super().__init__(service, decoder, entries, state.history[start:stop],
            (state.counts[start:stop], state.budgets[start:stop], state.deadlines[start:stop]))


class SharedRefillService(RollingService):
    wave_type = RefillWave

    def __init__(self, backend, shared, deadlines, *, settings=None, prompts=None):
        assert shared.runtime.cache_allocation == 'shared_refill_cache_arena_v1'
        block = shared.runtime.graph_cache_block_tokens
        required = shared.model.max_input_tokens + shared.generation.max_new_tokens
        capacity = (required + block - 1) // block * block
        arena = RefillCacheArena(backend.cache_config, shared.runtime.batch_size, capacity,
                                backend.model.lm_head.weight.dtype, backend.device)
        self.refill = RefillState(arena, shared.generation.max_new_tokens, backend.tokenizer.pad_token_id)
        super().__init__(backend, shared, deadlines, settings=settings, prompts=prompts)
        self.execution_metadata.update(cache_arena_bytes=arena.nbytes,
            refill_state_bytes=self.refill.nbytes, cache_arena_max_cache_tokens=capacity,
            cache_arena_max_batch_size=shared.runtime.batch_size,
            compaction_scratch_bytes=arena.compaction_scratch_bytes)

    def prefill(self, batch):
        backend = self.backend
        started = time.perf_counter()
        lengths = [len(request.input_ids) for request in batch]
        inputs = backend.tokenizer.pad({'input_ids': [request.input_ids for request in batch],
                    'attention_mask': [[1] * length for length in lengths]},
                    padding=True, return_tensors='pt').to(backend.device)
        record = dict(task_ids=[request.task_id for request in batch], batch_size=len(batch),
            messages=[request.messages for request in batch], input_tokens=lengths,
            requested_max_new_tokens=[request.max_tokens for request in batch],
            max_new_tokens=max(request.max_tokens for request in batch), temperature=self.shared.generation.temperature,
            queue_seconds=[started - request.submitted for request in batch],
            row_stops=[list(request.stop) for request in batch], stop=[],
            started_monotonic=started, active_token_slots=0, executed_token_slots=0,
            graph_capture_seconds=0.0, decode_engine='cuda_graph',
            cache_allocation=self.shared.runtime.cache_allocation,
            forward_input_shapes=[list(inputs['input_ids'].shape)],
            timing_scope='Overlapping admission cohorts; use complete run wall time.',
            output_rows=len(batch), batch_capacity=self.batch_size)
        for key in ('output_tokens', 'output_token_ids', 'texts', 'truncated', 'expired', 'finish_reasons',
                    'row_tail_wait_seconds', 'response_seconds', 'decode'):
            record[key] = [None] * len(batch)
        self.records.append(record)
        entries = [Entry(request, record, index) for index, request in enumerate(batch)]
        record['preparation_seconds'] = time.perf_counter() - started
        decoder, rows = self._prefill_state(batch, inputs)
        return self.wave_type(self, decoder, entries, rows.start, rows.stop)

    def _prefill_state(self, batch, inputs):
        return self.refill.prefill(self.backend, batch,
            [self.deadlines.end(request.task_id) for request in batch], inputs, self.graph_pool, self.graph_stream)

    def _replay(self, wave):
        wave.decoder.graph.replay()
        self.scheduling_records[-1]['steps'] += 1

    def bind(self, entries):
        size = len(entries)
        if size not in self.rolling:
            if len(self.rolling) == self.shared.runtime.graph_cache_size:
                self.rolling.popitem(last=False)
            self.rolling[size] = self.refill.decode(self.backend, self.graph_pool, self.graph_stream)
        self.rolling.move_to_end(size)
        decoder = self.rolling[size]
        capture = 0.0
        if decoder.graph is None:
            started = time.perf_counter()
            decoder.capture(self.shared.runtime.graph_warmup_steps)
            capture = time.perf_counter() - started
        entries[0].record['graph_capture_seconds'] += capture
        self.scheduling_records.append({'task_ids': [entry.request.task_id for entry in entries],
            'decode_input_shape': [size, 1], 'steps': 0, 'cache_capacity': decoder.capacity,
            'graph_capture_seconds': capture, 'started_monotonic': time.perf_counter()})
        return self.wave_type(self, decoder, entries, 0, size)

    def compact(self, entries, survivors):
        self.refill.compact(survivors)
        retained = [entries[index] for index in survivors]
        assert [entry.request for entry in retained] == self.refill.requests
        return retained

    @torch.inference_mode()
    def _serve(self):
        torch.cuda.synchronize(self.backend.device)
        torch.cuda.reset_peak_memory_stats(self.backend.device)
        self.origin_host = time.perf_counter()
        self.origin = torch.cuda.Event(enable_timing=True)
        self.origin.record()
        entries, keep, wave, closing = [], [], None, False
        while not closing or entries:
            if len(keep) != len(entries):
                entries = self.compact(entries, keep)
                wave = None
            batch = []
            if not entries and not closing:
                request = self.requests.get()
                if request is None:
                    break
                batch.append(request)
            deadline = time.perf_counter() + self.batch_wait_seconds
            while not closing and len(batch) + len(entries) < self.batch_size:
                try:
                    request = (self.requests.get(timeout=max(0, deadline - time.perf_counter()))
                               if not entries else self.requests.get_nowait())
                except Empty:
                    break
                if request is None:
                    closing = True
                else:
                    batch.append(request)
            try:
                if batch:
                    started = time.perf_counter()
                    batch = self._validate_inputs(batch)
                    validation = time.perf_counter() - started
                    if batch:
                        incoming = self.prefill(batch)
                        incoming.entries[0].record['input_validation_seconds'] = validation
                        new_keep = incoming.emit()
                        survivors = list(range(len(entries))) + [len(entries) + row for row in new_keep]
                        entries.extend(incoming.entries)
                        entries = self.compact(entries, survivors)
                        wave = None
                if not entries:
                    keep = []
                    continue
                if wave is None:
                    wave = self.bind(entries)
                self._replay(wave)
                keep = wave.emit()
            except Exception as error:
                for request in [entry.request for entry in entries] + batch:
                    if not request.future.done():
                        request.future.set_exception(error)
                while True:
                    request = self.requests.get()
                    if request is None:
                        return
                    if not request.future.done():
                        request.future.set_exception(error)


def select_service(config, service_factory):
    factory = BatchService if service_factory is None else service_factory
    constructor = factory.func if isinstance(factory, partial) else factory
    if config.runtime.generation_scheduling == 'cohort':
        return factory
    assert constructor is BatchService, 'Continuous scheduling requires the common autoregressive BatchService.'
    service = (SharedRefillService if config.runtime.cache_allocation == 'shared_refill_cache_arena_v1'
               else RollingService)
    return partial(service, *factory.args, **factory.keywords) if isinstance(factory, partial) else service
