from collections import OrderedDict
from concurrent.futures import Future
from copy import deepcopy
from dataclasses import dataclass
import time

import torch
from transformers import StoppingCriteria

from jev_spawn.service.generation import GenerationService, Request
from jev_spawn.service.runtime_contract import RuntimeContract
from jev_spawn.service.stops import StopCache
from jev_spawn.service.resources import ADAPTER_SETTINGS
from jev_spawn.runtime.cache_arena import StaticCacheArena
from jev_spawn.runtime.decoding import CapturedDecode


@dataclass
class BatchedRequest(Request):
    return_tokens: bool = False

    @property
    def signature(self):
        return self.temperature


class SampleStops(StoppingCriteria):
    def __init__(self, backend, batch, prompt_width, deadlines, stop_cache):
        self.prompt_width = prompt_width
        self.checked_monotonic = []
        self.eos = torch.tensor(backend.eos_ids, device=backend.device)
        self.lengths = torch.zeros(len(batch), dtype=torch.long, device=backend.device)
        self.causes = torch.zeros_like(self.lengths)
        self.budgets = torch.tensor([request.max_tokens for request in batch], device=backend.device)
        self.deadlines = torch.tensor([deadlines.end(request.task_id) for request in batch],
                                      device=backend.device, dtype=torch.float64)
        self.strings = []
        for stops in dict.fromkeys(request.stop for request in batch):
            if stops:
                criteria = stop_cache.get(stops)
                mask = torch.tensor([request.stop == stops for request in batch], device=backend.device)
                self.strings.append((criteria, mask))

    def __call__(self, input_ids, scores, **kwargs):
        now = time.perf_counter()
        self.checked_monotonic.append(now)
        natural = torch.isin(input_ids[:, -1], self.eos)
        for criteria, mask in self.strings:
            natural |= criteria(input_ids[:, self.prompt_width:], scores) & mask
        count = input_ids.shape[1] - self.prompt_width
        budget = self.budgets <= count
        expired = self.deadlines <= now
        done = natural | budget | expired
        cause = torch.where(expired, 3, torch.where(natural, 1, torch.where(budget, 2, 0)))
        self.causes = torch.where(self.lengths == 0, cause, self.causes)
        self.lengths = torch.where(done & (self.lengths == 0),
                                   count, self.lengths)
        return done


class BatchService(RuntimeContract, GenerationService):
    def __init__(self, backend, shared, deadlines, *, settings=None, prompts=None):
        self.shared, self.deadlines = shared, deadlines
        self.execution_metadata = {'text_decode_engine': shared.runtime.decode_engine,
                                   'generation_scheduling': shared.runtime.generation_scheduling,
                                   'cache_allocation': shared.runtime.cache_allocation}
        self.decoders = OrderedDict()
        self.graph_pool = torch.cuda.graph_pool_handle()
        self.graph_stream = torch.cuda.Stream(device=backend.device)
        self.execution_metadata['graph_memory'] = 'One shared temporary pool and capture stream per serialized service; persistent graph outputs use external buffers.'
        self.graph_stats = {}
        self.delivered = {}
        self.interleaving_finite = False
        self.timeout_deliveries = {}
        self.execution_metadata['deadline_delivery'] = 'Root request waits end at their own deadlines; in-flight cohort GPU work may continue.'
        self.stop_cache = StopCache(backend.tokenizer, backend.device, shared.runtime.stop_engine)
        for key, value in shared.backend().items():
            assert backend.config[key] == value, 'Shared backend setting differs: ' + key
        self.cache_arena = None
        if shared.runtime.cache_allocation == 'shared_static_cache_arena_v1':
            block = shared.runtime.graph_cache_block_tokens
            required = shared.model.max_input_tokens + shared.generation.max_new_tokens
            capacity = (required + block - 1) // block * block
            self.cache_arena = StaticCacheArena(backend.cache_config, shared.runtime.batch_size, capacity,
                                                backend.model.lm_head.weight.dtype, backend.device)
            torch.cuda.synchronize(backend.device)
            self.execution_metadata.update(cache_arena_bytes=self.cache_arena.nbytes,
                                           cache_arena_max_batch_size=shared.runtime.batch_size,
                                           cache_arena_max_cache_tokens=capacity)
        super().__init__(backend, shared.runtime.batch_size, shared.runtime.batch_wait_seconds)

    def complete(self, messages, max_tokens, temperature, n=ADAPTER_SETTINGS['completion']['samples'], stop=None, *, task_id, return_tokens=False):
        assert isinstance(n, int) and n > 0
        return self.complete_batch([messages] * n, max_tokens, temperature, stop,
                                   task_id=task_id, return_tokens=return_tokens)

    def complete_batch(self, message_batches, max_tokens, temperature, stop, *, task_id, return_tokens=False):
        assert 0 < max_tokens <= self.shared.generation.max_new_tokens
        assert temperature == self.shared.generation.temperature
        self.deadlines.remaining(task_id)
        stops = (stop,) if isinstance(stop, str) else tuple(stop or ())
        requests = [BatchedRequest(self.contract_messages(messages), max_tokens, temperature, stops, task_id,
                                   time.perf_counter(), Future(), return_tokens=return_tokens) for messages in message_batches]
        for request in requests:
            self.requests.put(request)
        try:
            outputs = [request.future.result(timeout=self.deadlines.remaining(task_id)) for request in requests]
            self.deadlines.remaining(task_id)
        except TimeoutError as error:
            self.timeout_deliveries[task_id] = time.perf_counter()
            raise TimeoutError('Complete sample deadline exceeded: ' + task_id) from error
        return outputs

    def _live_requests(self, batch):
        valid = []
        for request in batch:
            try:
                self.deadlines.remaining(request.task_id)
            except TimeoutError as error:
                request.future.set_exception(error)
            else:
                valid.append(request)
        return valid

    def _validate_inputs(self, batch):
        valid = self._live_requests(batch)
        return super()._validate_inputs(valid) if valid else []

    def _stopping(self, batch, prompt_width):
        self.stopping = SampleStops(self.backend, batch, prompt_width, self.deadlines, self.stop_cache)
        return self.stopping

    def make_decoder(self, size, capacity):
        return CapturedDecode(self.backend, size, capacity, arena=self.cache_arena,
                              graph_pool=self.graph_pool, graph_stream=self.graph_stream)

    def validate_forward_batch(self, size, requests):
        assert size == len(requests)

    def _generate_tokens(self, inputs, options, stopping):
        if self.shared.runtime.decode_engine == 'eager':
            return super()._generate_tokens(inputs, options, stopping)
        size = inputs['input_ids'].shape[0]
        required = inputs['input_ids'].shape[1] + options['max_new_tokens']
        block = self.shared.runtime.graph_cache_block_tokens
        capacity = (required + block - 1) // block * block
        key = (size, capacity)
        plan = {'key': key, 'resident': tuple(self.decoders), 'create': key not in self.decoders,
                'evict': key not in self.decoders and len(self.decoders) == self.shared.runtime.graph_cache_size}
        if self.shared.runtime.world_size > 1:
            commands = self.backend.parallel_commands
            plan = commands.leader_value(plan if commands.is_leader else None)
            assert plan['key'] == key and plan['resident'] == tuple(self.decoders)
        if plan['create']:
            if plan['evict']:
                self.decoders.popitem(last=False)
            self.decoders[key] = self.make_decoder(size, capacity)
        self.decoders.move_to_end(key)
        if self.shared.runtime.world_size > 1:
            capture = commands.leader_value(self.decoders[key].graph is None if commands.is_leader else None)
            assert capture == (self.decoders[key].graph is None)
        callback = self._deliver if self.shared.runtime.row_delivery == 'immediate' else None
        output, events, capture = self.decoders[key].generate_tokens(
            inputs, options, stopping, self.shared.runtime.graph_warmup_steps, callback, self._between_decode_steps)
        self.graph_stats = {'graph_capture_seconds': capture, 'graph_replays': len(events) - 1,
                            'graph_input_shape': [size, 1], 'resident_graphs': len(self.decoders),
                            'graph_cache_capacity': capacity, 'required_cache_tokens': required,
                            'cache_allocation': self.shared.runtime.cache_allocation}
        self.graph_stats['capture_state_memory'] = self.decoders[key].capture_memory
        return output, events

    def _between_decode_steps(self):
        pass

    def _deliver(self, tokens, indices):
        rows = torch.tensor(indices, device=self.backend.device)
        token_ids = tokens.index_select(0, rows).tolist()
        texts = self.backend.tokenizer.batch_decode(token_ids, skip_special_tokens=True)
        for index, text, ids in zip(indices, texts, token_ids):
            request = self.current_batch[index]
            positions = [text.find(stop) for stop in request.stop if stop in text]
            if positions:
                text = text[:min(positions)]
            self.delivered[index] = time.perf_counter()
            reason = {1: 'stop', 2: 'length', 3: 'timeout'}[int(self.stopping.causes[index])]
            request.future.set_result({'text': text, 'token_ids': ids, 'finish_reason': reason}
                                      if request.return_tokens else text)

    def _generate(self, batch):
        self.current_batch = batch
        self.delivered = {}
        self.interleaved_peak_allocated = 0
        self.interleaved_peak_reserved = 0
        shapes = []

        def observe(module, args, kwargs):
            if self.interleaving_finite:
                return
            ids, mask = kwargs['input_ids'], kwargs['attention_mask']
            inputs = kwargs['inputs_embeds'] if ids is None else ids
            assert inputs.ndim == (3 if ids is None else 2)
            shape = list(inputs.shape[:2])
            self.validate_forward_batch(shape[0], batch)
            mask_rows = mask['full_attention'].shape[0] if isinstance(mask, dict) else mask.shape[0]
            assert mask_rows == shape[0]
            shapes.append(shape)

        assert 1 <= len(batch) <= self.batch_size
        model = (self.backend.model.model.language_model if self.shared.runtime.decode_engine == 'cuda_graph'
                 else self.backend.model)
        hook = model.register_forward_pre_hook(observe, with_kwargs=True)
        try:
            texts = super()._generate(batch)
        finally:
            hook.remove()
        assert shapes and len(texts) == len(batch)
        record = batch[0].record
        record['messages'] = [request.messages for request in batch]
        record['output_texts'] = texts
        record['peak_allocated_bytes'] = max(record['peak_allocated_bytes'], self.interleaved_peak_allocated)
        record['peak_reserved_bytes'] = max(record['peak_reserved_bytes'], self.interleaved_peak_reserved)
        assert len(record['output_tokens']) == len(record['decode']) == len(batch)
        record['forward_input_shapes'] = shapes
        record['decode_engine'] = self.shared.runtime.decode_engine
        record.update(self.graph_stats)
        record['valid_output_mask'] = [True] * len(batch)
        record['output_rows'] = len(texts)
        record['row_delivery'] = self.shared.runtime.row_delivery
        if self.delivered:
            record['row_tail_wait_seconds'] = [
                max(0.0, self.delivered[index] - (record['finished_monotonic'] - tail))
                for index, tail in enumerate(record['row_tail_wait_seconds'])]
        record['batch_capacity'] = self.batch_size
        record['row_stops'] = [list(request.stop) for request in batch]
        causes = self.stopping.causes.tolist()
        record['finish_reasons'] = [{0: 'length', 1: 'stop', 2: 'length', 3: 'timeout'}[cause] for cause in causes]
        record['truncated'] = [cause != 1 for cause in causes]
        record['expired'] = [time.perf_counter() >= self.deadlines.end(request.task_id) for request in batch]
        record['row_deadlines_monotonic'] = [self.deadlines.end(request.task_id) for request in batch]
        record['row_timeout_delivery_monotonic'] = [self.timeout_deliveries.get(request.task_id) for request in batch]
        record['decode_stop_checks_monotonic'] = self.stopping.checked_monotonic
        record['decode_stop_checks_after_row_deadline'] = [sum(
            checked >= self.deadlines.end(request.task_id) for checked in self.stopping.checked_monotonic)
            for request in batch]
        record['deadline_work_semantics'] = ('Stop-check timestamps record host observation points for native '
            'cohort decode steps. Expired rows can remain in those batched forwards; no GPU cancellation claim.')
        return [{'text': text, 'token_ids': ids, 'finish_reason': reason} if request.return_tokens else text
                for request, text, ids, reason in zip(batch, texts, record['output_token_ids'],
                                                     record['finish_reasons'], strict=True)]
