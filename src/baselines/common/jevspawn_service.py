from concurrent.futures import Future
from dataclasses import dataclass, field as dataclass_field
from functools import lru_cache
from queue import Empty
import time

import torch

from baselines.common.service import BatchedRequest, BatchService
from jev_spawn.schema import CONTROLLER, controller_prefix, controller_prompts
from jev_spawn.algo.structured import common_prefix
from jev_spawn.infra.configuration import CORE
from jev_spawn.infra.readout_labels import AdmittedPrompt
from methods.program_execution.grouped import score_grouped
from baselines.common.context_window import truncate_prompt
from jev_spawn.infra.prompts import load_prompt
from jev_spawn.algo.composition import rank_extensions


class DecisionFuture(Future):
    def set_result(self, result):
        self.delivered_monotonic = time.perf_counter()
        super().set_result(result)


@dataclass
class DecisionRequest(BatchedRequest):
    field: dict = None
    admitted: AdmittedPrompt = dataclass_field(init=False)
    root_tokens: tuple = dataclass_field(init=False)

    @property
    def signature(self):
        return 'finite'


class StructuredService(BatchService):
    request_type = DecisionRequest

    def _admit_input(self, request, rendered, tokens):
        super()._admit_input(request, rendered, tokens)
        if isinstance(request, DecisionRequest):
            request.admitted = AdmittedPrompt(rendered, tuple(tokens))
            system = next(message['content'] for message in request.messages if message['role'] == 'system')
            prefix_tokens = self.root_prefix_tokens(system, request.field['context'])
            length = common_prefix([tokens, prefix_tokens])
            assert 0 < length < len(tokens)
            request.root_tokens = tuple(tokens[:length])

    def _root_prefix_tokens(self, system, context):
        prefix = controller_prefix(context, '')
        rendered, = self.backend._render([prefix], system)
        return tuple(self.backend.tokenizer(rendered, add_special_tokens=False)['input_ids'])

    def __init__(self, backend, shared, deadlines, *, settings, prompts):
        self.field_distributions = {}
        self.field_mode = settings['field_mode']
        self.finite_readout = settings['finite_readout']
        assert self.finite_readout == 'native_gpu_rows'
        self.finite_scheduling = settings['finite_scheduling']
        self.input_window = settings['input_window']
        self.input_notice = load_prompt(self.input_window['prompt'])
        assert self.finite_scheduling in {'cohort', 'decode_step'}
        assert self.finite_scheduling != 'decode_step' or shared.runtime.decode_engine == 'cuda_graph'
        CONTROLLER['option_template'] = prompts['option_template']
        super().__init__(backend, shared, deadlines, settings=settings, prompts=prompts)
        self.root_prefix_tokens = lru_cache(maxsize=shared.runtime.root_batch_size)(self._root_prefix_tokens)
        self.execution_metadata.update(finite_execution=self.field_mode, finite_scheduling=self.finite_scheduling,
                                       readout=self.finite_readout, candidate_labels=backend.answer_labels,
                                       grammar_decoding=False)

    def decide(self, fields, *, task_id):
        return self.enqueue_decisions(fields, task_id=task_id, request_type=self.request_type)

    def initial_scores(self):
        return self.backend.finite_output_weights.new_zeros((1,))

    def extend(self, identities, scores, *, task_id, width):
        distributions = [self.field_distributions.pop((task_id, identity)) for identity in identities]
        return rank_extensions(distributions, scores, width)

    def _validate_inputs(self, batch):
        batch = self._live_requests(batch)
        if not batch:
            return []
        tokenizer = self.backend.tokenizer
        rendered = tokenizer.apply_chat_template(
            [request.messages for request in batch], tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
        encoded = tokenizer(rendered, padding=False, truncation=False, add_special_tokens=False)
        valid = []
        for request, text, tokens in zip(batch, rendered, encoded['input_ids'], strict=True):
            effective, ids, metadata = truncate_prompt(
                tokenizer, text, tokens, self.backend.config['max_input_tokens'],
                self.input_window, self.input_notice)
            self._admit_input(request, effective, ids)
            request.input_window = metadata
            valid.append(request)
        return valid

    def enqueue_decisions(self, fields, *, task_id, request_type):
        ready = time.perf_counter()
        self.deadlines.remaining(task_id)
        now = time.perf_counter()
        direct = {}
        requests = []
        for field in fields:
            options = field['options']
            if len(options) == 1:
                option = options[0]
                direct[field['id']] = {
                    'id': field['id'], 'choice': option['id'],
                    'probabilities': [1.0], 'option_logits': [0.0],
                    'option_ids': [option['id']],
                    'ranked_option_ids': [option['id']],
                    'input_tokens': 0, 'ready_monotonic': ready, 'submitted_monotonic': now,
                    'delivered_monotonic': now,
                }
                self.records.append({'operation': 'finite_direct', 'task_ids': [task_id],
                    'node_ids': [field['id']], 'batch_size': 1, 'input_tokens': [0],
                    'output_tokens': [0], 'elapsed_seconds': 0.0, 'queue_seconds': [0.0],
                    'output_rows': 1, 'valid_output_mask': [True],
                    'structured': {'groups': [[direct[field['id']]]], 'batch_size': 1,
                                   'option_counts': [1], 'direct': True}})
                continue
            assert CORE['controller']['minimum_options'] <= len(options) <= len(self.backend.answer_labels), (
                f'Finite candidate count {len(options)} is outside native range '
                f'[{CORE["controller"]["minimum_options"]}, {len(self.backend.answer_labels)}].')
            history = field.get('history', '')
            prompt = controller_prompts(
                [field['state']], field['question'], options,
                list(self.backend.answer_labels[:len(options)]),
                CONTROLLER['output_instruction'], contexts=[field['context']],
                histories=[history]
            )[0]
            request = request_type(
                self.contract_messages([
                    {'role': 'system', 'content': CONTROLLER['system']}, {'role': 'user', 'content': prompt}]),
                1, self.shared.generation.temperature, (), task_id, time.perf_counter(), DecisionFuture(),
                field={**field, 'history': history})
            requests.append(request)
            self.requests.put(request)
        values = [request.future.result(timeout=self.deadlines.remaining(task_id)) for request in requests]
        model_values = [{**value, 'ready_monotonic': ready, 'submitted_monotonic': request.submitted,
                         'delivered_monotonic': request.future.delivered_monotonic}
                        for request, value in zip(requests, values, strict=True)]
        by_id = {value['id']: value for value in [*direct.values(), *model_values]}
        return [by_id[field['id']] for field in fields]

    def _between_decode_steps(self):
        if self.finite_scheduling == 'cohort':
            return
        for _ in range(self.requests.qsize()):
            try:
                request = self.requests.get_nowait()
            except Empty:
                break
            if request is None:
                self.requests.put(None)
                break
            self.pending.append(request)
        first = next((request for request in self.pending if isinstance(request, DecisionRequest)), None)
        if first is None:
            return
        capacity = min(self.batch_size, self.shared.runtime.branch_batch_size)
        batch = [request for request in self.pending if request.signature == first.signature][:capacity]
        self.pending = [request for request in self.pending if all(request is not selected for selected in batch)]
        started = time.perf_counter()
        original_count = len(batch)
        self.interleaved_peak_allocated = max(self.interleaved_peak_allocated,
                                             torch.cuda.max_memory_allocated(self.backend.device))
        self.interleaved_peak_reserved = max(self.interleaved_peak_reserved,
                                            torch.cuda.max_memory_reserved(self.backend.device))
        self.interleaving_finite = True
        try:
            valid = self._validate_inputs(batch)
            validation_seconds = time.perf_counter() - started
            if not valid:
                return
            values = self._generate(valid)
            record = valid[0].record
            record.update(input_validation_seconds=validation_seconds,
                          input_validation_batch_size=original_count, interleaved_at='decode_step')
            self.interleaved_peak_allocated = max(self.interleaved_peak_allocated, record['peak_allocated_bytes'])
            self.interleaved_peak_reserved = max(self.interleaved_peak_reserved, record['peak_reserved_bytes'])
            for request, value in zip(valid, values):
                request.future.set_result(value)
        except Exception as error:
            for request in batch:
                if not request.future.done():
                    request.future.set_exception(error)
            raise
        finally:
            self.interleaving_finite = False

    def _score(self, batch):
        fields = [{**request.field, 'id': str(index)} for index, request in enumerate(batch)]
        return score_grouped(self.backend, [fields], self.field_mode,
                             admitted_prompts=[request.admitted for request in batch])

    def _execution_trace(self, result, shapes):
        assert shapes
        return {'graph_capture_seconds': 0.0, 'decode_engine': 'finite_prefill'}

    @torch.inference_mode()
    def _generate(self, batch):
        if not isinstance(batch[0], DecisionRequest):
            return super()._generate(batch)
        shapes = []

        def observe(module, args, kwargs):
            shape = list(kwargs['input_ids'].shape)
            assert len(shape) == 2 and shape[0] <= self.shared.runtime.branch_batch_size
            shapes.append(shape)

        torch.cuda.synchronize(self.backend.device)
        torch.cuda.reset_peak_memory_stats(self.backend.device)
        started = time.perf_counter()
        hook = self.backend.model.model.register_forward_pre_hook(observe, with_kwargs=True)
        try:
            result = self._score(batch)
        finally:
            hook.remove()
        finished = time.perf_counter()
        values = result['groups'][0]
        distributions = result.pop('device_probabilities')
        for value, request, distribution in zip(values, batch, distributions, strict=True):
            value['id'] = request.field['id']
            value['options'] = request.field['options']
            self.field_distributions[request.task_id, request.field['id']] = distribution[:len(value['option_ids'])]
        assert len(values) == len(batch)
        trace = self._execution_trace(result, shapes)
        self._record_batch(batch, {'operation': 'finite', 'task_ids': [request.task_id for request in batch],
            'node_ids': [request.field['id'] for request in batch], 'batch_size': len(batch),
            'input_tokens': [value['input_tokens'] for value in values], 'output_tokens': [0] * len(batch),
            'decode': [], 'output_rows': len(values), 'forward_input_shapes': shapes,
            'valid_output_mask': [True] * len(batch), 'elapsed_seconds': finished - started,
            'started_monotonic': started, 'finished_monotonic': finished,
            'queue_seconds': [started - request.submitted for request in batch],
            'peak_allocated_bytes': result['peak_cuda_memory_bytes'],
            'peak_reserved_bytes': result['peak_cuda_reserved_bytes'],
            **trace, 'structured': result})
        return values
