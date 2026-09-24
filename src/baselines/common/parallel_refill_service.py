import json
import time
from types import SimpleNamespace

import torch

from baselines.common.parallel_service import decode_request, encode_request
from baselines.common.shared_refill_service import RefillWave, SharedRefillService
from jev_spawn.infra.configuration import ROOT
from jev_spawn.runtime.padded_rolling_decode import PaddedRollingDecode


SETTINGS = json.loads((ROOT / 'configs/baselines/common/parallel_refill_service.json').read_text())


class ParallelRefillWave(RefillWave):
    def __init__(self, service, decoder, entries, start, stop):
        super().__init__(service, decoder, entries, start, stop)
        self.start, self.stop = start, stop

    def emit(self):
        return self.service.emit_wave(self)


class ParallelRefillService(SharedRefillService):
    wave_type = ParallelRefillWave

    def __init__(self, backend, shared, deadlines, **kwargs):
        self.parallel_commands = backend.parallel_commands
        self.row_ids = []
        self.next_request_id = SETTINGS['initial_request_id']
        self.command_sequence = SETTINGS['initial_command_sequence']
        self.prefill_pending = None
        self.emitting_wave = None
        self.decode = None
        for operation in SETTINGS['commands']:
            self.parallel_commands.register(SETTINGS['commands'][operation], getattr(self, '_parallel_' + operation))
        super().__init__(backend, shared, deadlines, **kwargs)
        self.execution_metadata.update(host_scheduling='leader_rank_only',
            numerical_execution='tensor_parallel_continuous_refill', world_size=shared.runtime.world_size,
            stop_authority='leader_rank', prefill_scheduling='whole_prefill_between_decode_steps',
            physical_batch_buckets=SETTINGS['batch_buckets'],
            physical_decode_work='scheduling_records.steps times decode_input_shape[0]; active rows recorded separately',
            padding_outputs_excluded=True)

    def _serve(self):
        if self.parallel_commands.is_leader:
            return super()._serve()

    def dispatch(self, operation, payload):
        return self.parallel_commands.call(SETTINGS['commands'][operation],
            {**payload, 'sequence': self.command_sequence, 'live_ids': self.row_ids.copy()})

    def accept(self, payload):
        assert payload['sequence'] == self.command_sequence
        assert payload['live_ids'] == self.row_ids
        self.command_sequence += 1

    def _prefill_state(self, batch, inputs):
        self.prefill_pending = batch, inputs
        ids = list(range(self.next_request_id, self.next_request_id + len(batch)))
        self.next_request_id += len(batch)
        payload = {'requests': [encode_request(request) for request in batch], 'new_ids': ids,
            'started': {request.task_id: self.deadlines.started[request.task_id] for request in batch},
            'deadlines': [self.deadlines.end(request.task_id) for request in batch],
            'inputs': {name: {'shape': list(tensor.shape), 'dtype': str(tensor.dtype).removeprefix('torch.')}
                       for name, tensor in inputs.items()}}
        result = self.dispatch('prefill', payload)
        self.prefill_pending = None
        return result

    @torch.inference_mode()
    def _parallel_prefill(self, payload):
        self.accept(payload)
        self.deadlines.started.update(payload['started'])
        if self.parallel_commands.is_leader:
            batch, inputs = self.prefill_pending
        else:
            batch = [decode_request(record) for record in payload['requests']]
            inputs = {name: torch.empty(item['shape'], dtype=getattr(torch, item['dtype']), device=self.backend.device)
                      for name, item in payload['inputs'].items()}
        for tensor in inputs.values():
            self.parallel_commands.broadcast_tensor(tensor)
        decoder, rows = self.refill.prefill(self.backend, batch, payload['deadlines'], inputs,
                                           self.graph_pool, self.graph_stream)
        self.row_ids.extend(payload['new_ids'])
        return decoder, rows

    def compact(self, entries, survivors):
        self.dispatch('compact', {'survivors': survivors})
        retained = [entries[index] for index in survivors]
        assert [entry.request for entry in retained] == self.refill.requests
        return retained

    @torch.inference_mode()
    def _parallel_compact(self, payload):
        self.accept(payload)
        self.refill.compact(payload['survivors'])
        self.row_ids = [self.row_ids[index] for index in payload['survivors']]

    def bind(self, entries):
        size = len(entries)
        physical = next(bucket for bucket in SETTINGS['batch_buckets'] if bucket >= size)
        hit = physical in self.rolling
        evict = (next(iter(self.rolling)) if not hit and len(self.rolling) == self.shared.runtime.graph_cache_size
                 else None)
        capture = self.dispatch('bind', {'size': size, 'physical': physical, 'hit': hit, 'evict': evict,
            'capture': not hit or self.rolling[physical].graph is None})
        entries[0].record['graph_capture_seconds'] += capture
        self.scheduling_records.append({'task_ids': [entry.request.task_id for entry in entries],
            'request_ids': self.row_ids.copy(), 'decode_input_shape': [physical, 1],
            'active_rows': size, 'steps': 0,
            'cache_capacity': self.decode.capacity, 'graph_capture_seconds': capture,
            'started_monotonic': time.perf_counter()})
        view = SimpleNamespace(ids=self.decode.ids[:size], capacity=self.decode.capacity, graph=self.decode.graph)
        return self.wave_type(self, view, entries, 0, size)

    @torch.inference_mode()
    def _parallel_bind(self, payload):
        self.accept(payload)
        size, physical = payload['size'], payload['physical']
        assert (physical in self.rolling) == payload['hit']
        if payload['evict'] is not None:
            assert next(iter(self.rolling)) == payload['evict']
            self.rolling.pop(payload['evict'])
        rows = self.refill.arena.padded_decode_view(physical)
        if not payload['hit']:
            decoder = PaddedRollingDecode.from_rows(self.backend, rows, self.refill.key_positions,
                                                    self.graph_pool, self.graph_stream)
            decoder.configure_rows(size)
            self.rolling[physical] = decoder
        self.rolling.move_to_end(physical)
        self.decode = self.rolling[physical]
        self.decode.active_count.fill_(size)
        assert (self.decode.graph is None) == payload['capture']
        started = time.perf_counter()
        if payload['capture']:
            self.decode.capture(self.shared.runtime.graph_warmup_steps)
        return time.perf_counter() - started if payload['capture'] else 0.0

    def _replay(self, wave):
        self.dispatch('step', {})
        self.scheduling_records[-1]['steps'] += 1

    @torch.inference_mode()
    def _parallel_step(self, payload):
        self.accept(payload)
        self.decode.graph.replay()

    def emit_wave(self, wave):
        self.emitting_wave = wave
        result = self.dispatch('emit', {'start': wave.start, 'stop': wave.stop})
        self.emitting_wave = None
        return result

    @torch.inference_mode()
    def _parallel_emit(self, payload):
        self.accept(payload)
        if self.parallel_commands.is_leader:
            return super(ParallelRefillWave, self.emitting_wave).emit()
        self.refill.emit(payload['start'], payload['stop'])

    def close(self):
        super().close()
        self.decode = None
        self.rolling.clear()
        self.decoders.clear()
        self.refill = None
        self.cache_arena = None
