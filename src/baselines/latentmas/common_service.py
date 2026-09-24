from collections import OrderedDict
from concurrent.futures import Future
from functools import partial
from dataclasses import dataclass, field
import subprocess
import time
from types import SimpleNamespace

import torch

from baselines.common.errors import InputLimitError
from baselines.common.resources import ADAPTER_SETTINGS
from baselines.common.runtime_contract import RuntimeContract
from baselines.common.service import SampleStops
from baselines.common.stops import StopCache
from baselines.common.tasks import read
from baselines.latentmas.adapter import AGENTS, PROMPTS, SOURCE, HybridTransport, ModelAdapter
from baselines.latentmas.cache_compaction import compact_attention_history
from baselines.latentmas.native_hybrid_transport import CapturedPaddingTransport, ChunkedHybridTransport
from baselines.official.model_service import GenerationService, Request
from jev_spawn.runtime.cache_arena import StaticCacheArena
from jev_spawn.runtime.decoding import CapturedDecode


@dataclass
class LatentRequest(Request):
    role_ids: list = field(default_factory=list)
    role_messages: list = field(default_factory=list)


def role_messages(messages, prompts, model_path):
    question = '\n\n'.join(message['role'] + ':\n' + message['content'] for message in messages)
    args = SimpleNamespace(model_name=model_path)
    latent = [PROMPTS.build_agent_message_sequential_latent_mas(
        role=agent.role, question=question, context='', method='latent_mas', args=args)
        for agent in AGENTS.default_agents()[:-1]]
    return latent + [[{'role': 'system', 'content': prompts['judger']},
                      {'role': 'user', 'content': question}]]


class LatentTransport(HybridTransport):
    def __init__(self, backend, initial_cache, before_forward):
        super().__init__(backend)
        self.initial_cache = initial_cache
        self.before_forward = before_forward

    def _forward(self, embeddings, mask, cache):
        self.before_forward()
        cache = self.initial_cache if cache is None else cache
        return super()._forward(embeddings, mask, cache)


class SegmentedLatentTransport(ChunkedHybridTransport):
    def __init__(self, backend, initial_cache, before_forward):
        super().__init__(backend)
        self.initial_cache = initial_cache
        self.before_forward = before_forward

    def _forward(self, embeddings, mask, cache):
        self.before_forward()
        cache = self.initial_cache if cache is None else cache
        return super()._forward(embeddings, mask, cache)


class LatentDecode(CapturedDecode):
    def __init__(self, backend, batch_size, capacity, arena=None, graph_pool=None, graph_stream=None):
        super().__init__(backend, batch_size, capacity, arena=arena, graph_pool=graph_pool, graph_stream=graph_stream)
        self.padding_graph = None

    def initialize(self, transport):
        assert transport.cache is self.cache
        if self.backend.config['execution'] == 'qwen35_optimized':
            transport.mask = compact_attention_history(self.cache, transport.mask)
        width = transport.mask.shape[1]
        self.key_valid.fill_(True)
        self.key_valid[:, :width].copy_(transport.mask)
        self.positions.copy_(transport.mask.sum(1)[:, None])
        self.logits.copy_(self.backend.model.lm_head(transport.last_hidden))
        self.ids.copy_(self.logits.argmax(-1)[:, None])

    def prefill(self, inputs):
        return self.ids[:, 0].clone()


class LatentMASService(RuntimeContract, GenerationService):
    def __init__(self, backend, shared, deadlines, *, settings, prompts):
        assert shared.runtime.generation_scheduling == 'cohort'
        assert shared.runtime.decode_engine == 'cuda_graph'
        assert shared.generation.temperature == 0
        assert settings['memory_mode'] == 'full' and settings['alignment'] is True
        assert settings['latent_steps'] > 0
        assert settings['padding_transport'] in {'eager', 'cuda_graph_single_token', 'native_segments'}
        assert settings['padding_workspace_reserve_bytes'] > 0
        for key, value in shared.backend().items():
            assert backend.config[key] == value, 'Shared backend setting differs: ' + key
        self.shared, self.deadlines, self.settings, self.prompts = shared, deadlines, settings, prompts
        self.agents = AGENTS.default_agents()
        self.latent_role_count = sum(agent.role != 'judger' for agent in self.agents)
        self.decoders = OrderedDict()
        self.graph_pool = torch.cuda.graph_pool_handle()
        self.graph_stream = torch.cuda.Stream(device=backend.device)
        self.stop_cache = StopCache(backend.tokenizer, backend.device, shared.runtime.stop_engine)
        args = SimpleNamespace(latent_space_realign=True)
        if backend.config['world_size'] > 1:
            args.alignment_settings = read(settings['alignment_config'])
        self.wrapper = ModelAdapter(backend, args)
        started = time.perf_counter()
        self.alignment = self.wrapper._ensure_latent_realign_matrix(self.wrapper.model, backend.device, args)
        torch.cuda.synchronize(backend.device)
        self.alignment_seconds = time.perf_counter() - started
        self.scheduling_records = []
        self.timeout_deliveries = {}
        self.execution_metadata = {
            'generation_scheduling': 'cohort', 'text_decode_engine': 'cuda_graph',
            'graph_memory': 'One shared temporary pool and capture stream per serialized service; persistent graph outputs use external buffers.',
            'latent_decode_engine': 'eager', 'latent_steps_per_role': settings['latent_steps'],
            'roles': [agent.role for agent in self.agents],
            'cache_transport': 'one native static hybrid cache across every latent role and judger',
            'padding_transport': settings['padding_transport'],
            'text_cache_layout': ('compacted_valid_history' if backend.config['execution'] == 'qwen35_optimized'
                                  else 'original_role_history'),
            'deadline_delivery': 'Each root wait ends at its own sample deadline. In-flight cohort GPU work may continue and is reported explicitly.',
            'alignment_seconds': self.alignment_seconds,
            'upstream_revision': subprocess.check_output(
                ['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip(),
            'adaptation': 'Generic tool action judger; repeat full upstream latent collaboration after each tool observation.'}
        self.cache_arena = None
        self.execution_metadata['cache_allocation'] = shared.runtime.cache_allocation
        if shared.runtime.cache_allocation == 'shared_static_cache_arena_v1':
            block = shared.runtime.graph_cache_block_tokens
            required = shared.model.max_input_tokens + shared.generation.max_new_tokens
            capacity = (required + block - 1) // block * block
            self.cache_arena = StaticCacheArena(backend.cache_config, shared.runtime.batch_size, capacity,
                                                backend.model.lm_head.weight.dtype, backend.device)
            torch.cuda.synchronize(backend.device)
            self.execution_metadata.update(cache_arena_bytes=self.cache_arena.nbytes,
                cache_arena_max_batch_size=shared.runtime.batch_size, cache_arena_max_cache_tokens=capacity)
        super().__init__(backend, shared.runtime.batch_size, shared.runtime.batch_wait_seconds)

    def complete(self, messages, max_tokens, temperature,
                 n=ADAPTER_SETTINGS['completion']['samples'], stop=None, *, task_id):
        assert n == 1 and stop is None
        assert 0 < max_tokens <= self.shared.generation.max_new_tokens
        assert temperature == self.shared.generation.temperature
        self.deadlines.remaining(task_id)
        request = LatentRequest(self.contract_messages(messages), max_tokens, temperature, (), task_id,
                                time.perf_counter(), Future())
        self.requests.put(request)
        try:
            result = request.future.result(timeout=self.deadlines.remaining(task_id))
            self.deadlines.remaining(task_id)
        except TimeoutError as error:
            self.timeout_deliveries[task_id] = time.perf_counter()
            raise TimeoutError('Complete sample deadline exceeded: ' + task_id) from error
        return [result]

    def _reject_context(self, request, length):
        error = InputLimitError(f'Full LatentMAS role input history requires {length} '
                                f'tokens; context limit is {self.shared.model.max_input_tokens}. No truncation.')
        self.input_failures.append({'task_id': request.task_id, 'messages': request.messages,
            'role_input_tokens': list(map(len, request.role_ids)), 'required_context_tokens': length,
            'max_input_tokens': self.shared.model.max_input_tokens, 'error': str(error)})
        request.future.set_exception(error)

    def _validate_inputs(self, batch):
        valid = []
        for request in batch:
            try:
                self.deadlines.remaining(request.task_id)
            except TimeoutError as error:
                request.future.set_exception(error)
                continue
            request.role_messages = role_messages(request.messages, self.prompts, self.shared.model.path)
            rendered = [self.wrapper.render_chat(messages) for messages in request.role_messages]
            request.role_ids = self.backend.tokenizer(rendered, add_special_tokens=False,
                                                       truncation=False)['input_ids']
            request.input_ids = request.role_ids[0]
            required = sum(map(len, request.role_ids)) + self.latent_role_count * self.settings['latent_steps']
            if required > self.shared.model.max_input_tokens:
                self._reject_context(request, required)
            else:
                valid.append(request)
        if valid:
            physical = sum(max(len(request.role_ids[role]) for request in valid) for role in range(len(self.agents)))
            physical += self.latent_role_count * self.settings['latent_steps']
            if physical > self.shared.model.max_input_tokens:
                for request in valid:
                    self._reject_context(request, physical)
                return []
        return valid

    def _decoder(self, size, capacity):
        key = (size, capacity)
        create = key not in self.decoders
        evict = (next(iter(self.decoders))
                 if create and len(self.decoders) == self.shared.runtime.graph_cache_size else None)
        capture = create or self.decoders[key].graph is None
        if self.backend.config['world_size'] > 1:
            commands = self.backend.parallel_commands
            plan = commands.leader_value((create, evict, capture) if commands.is_leader else None)
            assert plan == (create, evict, capture), 'Tensor-parallel latent decoder cache states diverged.'
            create, evict, capture = plan
        if create:
            if evict is not None:
                self.decoders.pop(evict)
            self.decoders[key] = LatentDecode(self.backend, size, capacity, arena=self.cache_arena,
                graph_pool=self.graph_pool, graph_stream=self.graph_stream)
        self.decoders.move_to_end(key)
        decoder = self.decoders[key]
        decoder.cache.reset()
        return decoder

    def _check_deadlines(self, identities):
        decision = None
        if self.backend.config['world_size'] == 1 or self.backend.parallel_commands.is_leader:
            now = time.perf_counter()
            expired = [identity for identity in identities if now >= self.deadlines.end(identity)]
            decision = now, expired, len(expired) != len(identities)
        if self.backend.config['world_size'] > 1:
            decision = self.backend.parallel_commands.leader_value(decision)
        now, expired, dispatch = decision
        self.forward_deadline_checks.append({'checked_monotonic': now,
            'expired_task_ids': expired, 'forward_dispatched': dispatch,
            'batch_size': len(identities)})
        if not dispatch:
            raise TimeoutError('Every row in the latent batch exceeded its sample deadline.')

    def _stopping(self, batch, width):
        return SampleStops(self.backend, batch, width, self.deadlines, self.stop_cache)

    def _deadline_record(self, batch):
        return {'row_deadlines_monotonic': [self.deadlines.end(r.task_id) for r in batch],
                'row_timeout_delivery_monotonic': [self.timeout_deliveries.get(r.task_id) for r in batch],
                'forward_deadline_checks': self.forward_deadline_checks,
                'forward_dispatches_retaining_expired_row': [sum(
                    check['forward_dispatched'] and r.task_id in check['expired_task_ids']
                    for check in self.forward_deadline_checks) for r in batch],
                'deadline_work_semantics': 'Timeout releases the root immediately. An expired row can remain in native batched GPU forwards while live rows continue; no claim that its GPU work ceased at the deadline.'}


    @torch.inference_mode()
    def _generate(self, batch):
        try:
            return self._generate_batch(batch)
        except TimeoutError:
            self.records.append({'task_ids': [r.task_id for r in batch], 'batch_size': len(batch),
                'status': 'timeout', 'latent_forward_calls': self.wrapper.model.records,
                **self._deadline_record(batch),
                'latent_role_seconds': self.wrapper.role_times,
                'latent_steps_per_role': self.settings['latent_steps']})
            raise

    def _generate_batch(self, batch):
        backend, wrapper = self.backend, self.wrapper
        torch.cuda.synchronize(backend.device)
        torch.cuda.reset_peak_memory_stats(backend.device)
        started = time.perf_counter()
        self.forward_deadline_checks = []
        required = sum(max(len(request.role_ids[role]) for request in batch) for role in range(len(self.agents)))
        required += self.latent_role_count * self.settings['latent_steps'] + max(request.max_tokens for request in batch)
        block = self.shared.runtime.graph_cache_block_tokens
        capacity = (required + block - 1) // block * block
        decoder = self._decoder(len(batch), capacity)
        wrapper.reset()
        identities = [r.task_id for r in batch]
        if self.settings['padding_transport'] == 'eager':
            wrapper.model = LatentTransport(backend, decoder.cache, partial(self._check_deadlines, identities))
        elif self.settings['padding_transport'] == 'native_segments':
            wrapper.model = SegmentedLatentTransport(backend, decoder.cache, partial(self._check_deadlines, identities))
        else:
            wrapper.model = CapturedPaddingTransport(backend, decoder, self.shared.runtime.graph_warmup_steps,
                self.settings['padding_workspace_reserve_bytes'], partial(self._check_deadlines, identities))
        wrapper._latent_realign_matrices = {id(wrapper.model): self.alignment}
        role_widths = []
        for role, agent in enumerate(self.agents):
            ids = [request.role_ids[role] for request in batch]
            inputs = backend.tokenizer.pad({'input_ids': ids, 'attention_mask': [[1] * len(row) for row in ids]},
                                            padding=True, return_tensors='pt').to(backend.device)
            role_widths.append(inputs['input_ids'].shape[1])
            if agent.role != 'judger':
                wrapper.generate_latent_batch(**inputs, latent_steps=self.settings['latent_steps'],
                                              past_key_values=wrapper.model.cache)
            else:
                wrapper.model.phase = 'judger_prefill'
                wrapper.model(**inputs, past_key_values=wrapper.model.cache)
        decoder.initialize(wrapper.model)
        stopping = self._stopping(batch, inputs['input_ids'].shape[1])
        options = {'max_new_tokens': max(request.max_tokens for request in batch), 'do_sample': False,
                   'pad_token_id': backend.tokenizer.pad_token_id}
        delivered = {}

        def deliver(tokens, indices):
            if backend.config['world_size'] > 1 and not backend.parallel_commands.is_leader:
                return
            rows = torch.tensor(indices, device=backend.device)
            texts = backend.tokenizer.batch_decode(tokens.index_select(0, rows).tolist(), skip_special_tokens=True)
            for index, text in zip(indices, texts):
                delivered[index] = time.perf_counter()
                batch[index].future.set_result(text)

        callback = deliver if self.shared.runtime.row_delivery == 'immediate' else None
        origin = torch.cuda.Event(enable_timing=True)
        origin.record()
        decode_started = time.perf_counter()
        sequences, events, capture = decoder.generate_tokens(
            inputs, options, stopping, self.shared.runtime.graph_warmup_steps, callback)
        if backend.config['world_size'] > 1 and not backend.parallel_commands.is_leader:
            return
        generated = sequences[:, inputs['input_ids'].shape[1]:]
        counts = stopping.lengths.tolist()
        assert all(count > 0 for count in counts)
        ids = [row[:count] for row, count in zip(generated.tolist(), counts)]
        texts = backend.tokenizer.batch_decode(ids, skip_special_tokens=True)
        torch.cuda.synchronize(backend.device)
        finished = time.perf_counter()
        times = [decode_started + origin.elapsed_time(event) / 1000 for event in events]
        causes = stopping.causes.tolist()
        self.records.append({
            'task_ids': [r.task_id for r in batch], 'batch_size': len(batch), 'batch_capacity': self.batch_size,
            **self._deadline_record(batch),
            'messages': [r.messages for r in batch], 'role_messages': [r.role_messages for r in batch],
            'role_input_tokens': [list(map(len, r.role_ids)) for r in batch], 'role_padded_widths': role_widths,
            'input_tokens': [sum(map(len, r.role_ids)) for r in batch], 'output_tokens': counts,
            'output_token_ids': ids, 'texts': texts, 'max_new_tokens': options['max_new_tokens'],
            'temperature': batch[0].temperature, 'requested_max_new_tokens': [r.max_tokens for r in batch],
            'truncated': [cause != 1 for cause in causes],
            'finish_reasons': [{1: 'stop', 2: 'length', 3: 'timeout'}[cause] for cause in causes],
            'expired': [finished >= self.deadlines.end(r.task_id) for r in batch],
            'queue_seconds': [started - r.submitted for r in batch], 'elapsed_seconds': finished - started,
            'started_monotonic': started, 'finished_monotonic': finished,
            'latent_role_seconds': wrapper.role_times, 'latent_steps_per_role': self.settings['latent_steps'],
            'latent_forward_calls': wrapper.model.records,
            'forward_input_shapes': [[r['batch_size'], r['tokens']] for r in wrapper.model.records],
            'cache_allocation': self.shared.runtime.cache_allocation,
            'decode_engine': 'cuda_graph', 'graph_capture_seconds': capture, 'graph_replays': len(events) - 1,
            'capture_state_memory': decoder.capture_memory,
            'graph_input_shape': [len(batch), 1], 'graph_cache_capacity': decoder.capacity,
            'decode_steps': len(events), 'active_token_slots': sum(counts),
            'executed_token_slots': len(batch) * generated.shape[1],
            'latent_token_slots': len(batch) * self.latent_role_count * self.settings['latent_steps'],
            'valid_output_mask': [True] * len(batch), 'output_rows': len(batch),
            'row_delivery': self.shared.runtime.row_delivery,
            'row_tail_wait_seconds': [max(0.0, delivered.get(index, finished) - times[count - 1])
                                      for index, count in enumerate(counts)],
            'decode': [{'ttft_seconds': times[0] - started,
                        'inter_token_seconds': [b - a for a, b in zip(times[:count], times[1:count])]}
                       for count in counts],
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(backend.device),
            'peak_reserved_bytes': torch.cuda.max_memory_reserved(backend.device)})
        return texts
