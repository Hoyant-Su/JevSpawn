from collections import OrderedDict
from dataclasses import dataclass, field
from queue import Empty
import time

import torch

from baselines.common.service import BatchService
from jev_spawn.runtime.decoding import CapturedDecode
from jev_spawn.runtime.rolling_decode import RollingDecode


@dataclass
class Entry:
    request: object
    record: dict
    row: int
    events: list = field(default_factory=list)
    count: int = 0


class Wave:
    def __init__(self, service, decoder, entries, history, resident_state=None):
        self.service, self.decoder, self.entries, self.history = service, decoder, entries, history
        device = service.backend.device
        if resident_state is None:
            self.counts = torch.tensor([entry.count for entry in entries], device=device)
            self.budgets = torch.tensor([entry.request.max_tokens for entry in entries], device=device)
            self.deadlines = torch.tensor([service.deadlines.end(entry.request.task_id) for entry in entries],
                                          device=device, dtype=torch.float64)
        else:
            self.counts, self.budgets, self.deadlines = resident_state
        self.strings = []
        for stops in dict.fromkeys(entry.request.stop for entry in entries):
            if stops:
                criteria = service.stop_cache.get(stops)
                offsets = torch.arange(criteria.maximum_token_len, device=device)
                mask = torch.tensor([entry.request.stop == stops for entry in entries], device=device)
                self.strings.append((criteria, offsets, mask))

    def emit(self):
        service, backend = self.service, self.service.backend
        tokens = self.decoder.ids[:, 0]
        self.history.scatter_(1, self.counts[:, None], tokens[:, None])
        self.counts.add_(1)
        natural = torch.isin(tokens, service.eos)
        for criteria, offsets, mask in self.strings:
            positions = self.counts[:, None] - len(offsets) + offsets
            window = self.history.gather(1, positions.clamp_min(0))
            window.masked_fill_(positions < 0, backend.tokenizer.pad_token_id)
            natural |= criteria(window, None) & mask
        cause = torch.where(self.deadlines <= time.perf_counter(), 3,
                            torch.where(natural, 1, torch.where(self.counts >= self.budgets, 2, 0)))
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        causes = cause.tolist()
        for entry in self.entries:
            entry.count += 1
            entry.events.append(event)
        finished = [index for index, value in enumerate(causes) if value]
        if finished:
            indices = torch.tensor(finished, device=backend.device)
            outputs = self.history.index_select(0, indices).tolist()
            outputs = [tokens[:self.entries[index].count] for index, tokens in zip(finished, outputs)]
            texts = backend.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            for index, ids, text in zip(finished, outputs, texts):
                entry = self.entries[index]
                positions = [text.find(stop) for stop in entry.request.stop if stop in text]
                if positions:
                    text = text[:min(positions)]
                service.finish(entry, causes[index], ids, text)
        return [index for index, value in enumerate(causes) if not value]


class RollingService(BatchService):
    def __init__(self, backend, shared, deadlines, *, settings=None, prompts=None):
        assert shared.runtime.decode_engine == 'cuda_graph'
        assert shared.generation.temperature == 0
        self.rolling = OrderedDict()
        self.scheduling_records = []
        self.eos = torch.tensor(backend.eos_ids, device=backend.device)
        super().__init__(backend, shared, deadlines, settings=settings, prompts=prompts)

    def finish(self, entry, cause, ids, text):
        record, row = entry.record, entry.row
        now = time.perf_counter()
        times = [self.origin_host + self.origin.elapsed_time(event) / 1000 for event in entry.events]
        values = dict(output_tokens=entry.count, output_token_ids=ids, texts=text,
                      truncated=cause != 1, expired=cause == 3,
                      finish_reasons={1: 'stop', 2: 'length', 3: 'timeout'}[cause],
                      row_tail_wait_seconds=now - times[-1],
                      response_seconds=now - entry.request.submitted,
                      decode={'ttft_seconds': times[0] - record['started_monotonic'],
                              'inter_token_seconds': [b - a for a, b in zip(times, times[1:])]})
        for key, value in values.items():
            record[key][row] = value
        record['active_token_slots'] += entry.count
        record['executed_token_slots'] += entry.count
        record['elapsed_seconds'] = now - record['started_monotonic']
        record['finished_monotonic'] = now
        record['peak_allocated_bytes'] = torch.cuda.max_memory_allocated(self.backend.device)
        record['peak_reserved_bytes'] = torch.cuda.max_memory_reserved(self.backend.device)
        entry.request.future.set_result({'text': text, 'token_ids': ids,
                                         'finish_reason': values['finish_reasons']}
                                        if entry.request.return_tokens else text)

    def prefill(self, batch):
        backend = self.backend
        started = time.perf_counter()
        lengths = [len(request.input_ids) for request in batch]
        inputs = backend.tokenizer.pad({'input_ids': [request.input_ids for request in batch],
                    'attention_mask': [[1] * length for length in lengths]},
                    padding=True, return_tensors='pt').to(backend.device)
        width = inputs['input_ids'].shape[1]
        block = self.shared.runtime.graph_cache_block_tokens
        capacity = (width + max(request.max_tokens for request in batch) + block - 1) // block * block
        decoder = CapturedDecode(backend, len(batch), capacity)
        record = dict(task_ids=[request.task_id for request in batch], batch_size=len(batch),
                      messages=[request.messages for request in batch], input_tokens=lengths,
                      requested_max_new_tokens=[request.max_tokens for request in batch],
                      max_new_tokens=max(request.max_tokens for request in batch), temperature=self.shared.generation.temperature,
                      queue_seconds=[started - request.submitted for request in batch],
                      row_stops=[list(request.stop) for request in batch], stop=[],
                      started_monotonic=started, active_token_slots=0, executed_token_slots=0,
                      graph_capture_seconds=0.0, decode_engine='cuda_graph',
                      forward_input_shapes=[list(inputs['input_ids'].shape)],
                      timing_scope='Overlapping admission cohorts; use complete run wall time.',
                      output_rows=len(batch), batch_capacity=self.batch_size)
        for key in ('output_tokens', 'output_token_ids', 'texts', 'truncated', 'expired', 'finish_reasons',
                    'row_tail_wait_seconds', 'response_seconds', 'decode'):
            record[key] = [None] * len(batch)
        self.records.append(record)
        entries = [Entry(request, record, index) for index, request in enumerate(batch)]
        history = torch.full((len(batch), self.shared.generation.max_new_tokens), backend.tokenizer.pad_token_id,
                             device=backend.device, dtype=torch.long)
        wave = Wave(self, decoder, entries, history)
        record['preparation_seconds'] = time.perf_counter() - started
        decoder.prefill(inputs)
        return wave

    def merge(self, sources):
        entries = [wave.entries[index] for wave, rows in sources for index in rows]
        device = self.backend.device
        indexed = [(wave, torch.tensor(rows, device=device)) for wave, rows in sources]
        capacity = max(wave.decoder.capacity for wave, _ in sources)
        key = (len(entries), capacity)
        if key not in self.rolling:
            if len(self.rolling) == self.shared.runtime.graph_cache_size:
                self.rolling.popitem(last=False)
            self.rolling[key] = RollingDecode(self.backend, len(entries), capacity, sources[0][0].decoder)
        self.rolling.move_to_end(key)
        decoder = self.rolling[key]
        decoder.load([(wave.decoder, rows) for wave, rows in indexed])
        history = torch.cat([wave.history.index_select(0, rows) for wave, rows in indexed])
        wave = Wave(self, decoder, entries, history)
        capture = 0.0
        if decoder.graph is None:
            started = time.perf_counter()
            decoder.capture(self.shared.runtime.graph_warmup_steps)
            capture = time.perf_counter() - started
        entries[0].record['graph_capture_seconds'] += capture
        self.scheduling_records.append({'task_ids': [entry.request.task_id for entry in entries],
            'decode_input_shape': [len(entries), 1], 'steps': 0, 'cache_capacity': capacity,
            'graph_capture_seconds': capture, 'started_monotonic': time.perf_counter()})
        return wave

    @torch.inference_mode()
    def _serve(self):
        torch.cuda.synchronize(self.backend.device)
        torch.cuda.reset_peak_memory_stats(self.backend.device)
        self.origin_host = time.perf_counter()
        self.origin = torch.cuda.Event(enable_timing=True)
        self.origin.record()
        active, keep, closing = None, [], False
        while not closing or active is not None:
            batch = []
            if active is None and not closing:
                first = self.requests.get()
                if first is None:
                    break
                batch.append(first)
            deadline = time.perf_counter() + self.batch_wait_seconds
            while not closing and len(batch) + len(keep) < self.batch_size:
                try:
                    request = (self.requests.get(timeout=max(0, deadline - time.perf_counter()))
                               if active is None else self.requests.get_nowait())
                except Empty:
                    break
                if request is None:
                    closing = True
                else:
                    batch.append(request)
            incoming = None
            try:
                if batch:
                    started = time.perf_counter()
                    batch = self._validate_inputs(batch)
                    validation = time.perf_counter() - started
                    if batch:
                        incoming = self.prefill(batch)
                        incoming.entries[0].record['input_validation_seconds'] = validation
                        new_keep = incoming.emit()
                sources = [(active, keep)] if keep else []
                if incoming is not None and new_keep:
                    sources.append((incoming, new_keep))
                if not sources:
                    active, keep = None, []
                    continue
                if incoming is not None or len(keep) != len(active.entries):
                    active = self.merge(sources)
                active.decoder.graph.replay()
                self.scheduling_records[-1]['steps'] += 1
                keep = active.emit()
                if not keep:
                    active = None
            except Exception as error:
                entries = [] if active is None else active.entries
                requests = [entry.request for entry in entries] + batch
                for request in requests:
                    if not request.future.done():
                        request.future.set_exception(error)
                # An infrastructure failure invalidates this service; every queued caller is released with the error.
                while True:
                    request = self.requests.get()
                    if request is None:
                        return
                    request.future.set_exception(error)
