import ast
import inspect
import textwrap
import time

import torch

from baselines.common.graph_finite_service import StableGraphFiniteService
from baselines.common.history_service import HistoryBatchService
from baselines.common.service import BatchService
from jev_spawn.algo.structured import padded
from jev_spawn.infra.history_prefill import HistoryDecode
from jev_spawn.runtime.decoding import CapturedDecode
from jev_spawn.runtime.native_cache_batch import split_native_cache
from jev_spawn.runtime.native_restore import load_native_prefixes
from tests.infra.history_cache.checkpoints import CheckpointHistory


class TerminalCheckpointHistory(CheckpointHistory):
    @torch.inference_mode()
    def compute_terminal(self, sequences):
        started = time.perf_counter()
        policy = self.settings
        unique = list(dict.fromkeys(map(tuple, sequences)))
        complete = [sequence in self.cache.outputs and self.cache.outputs[sequence] is not None for sequence in unique]
        prefixes = [sequence if hit else self.cache.prefix(sequence[:policy['exclude_last']])
                    for sequence, hit in zip(unique, complete, strict=True)]
        plan = (prefixes, complete, len(self.cache.entries), self.cache.bytes)
        commands = self.backend.parallel_commands
        assert plan == commands.leader_value(plan if commands.is_leader else None)
        states = {sequence: self.cache.read(prefix) for sequence, prefix in zip(unique, prefixes, strict=True) if prefix}
        logits = {sequence: self.cache.outputs[sequence] for sequence, hit in zip(unique, complete, strict=True) if hit}
        cold = [sequence for sequence, prefix in zip(unique, prefixes, strict=True) if not prefix]
        warm = [(sequence, prefix) for sequence, prefix, hit in zip(unique, prefixes, complete, strict=True) if prefix and not hit]
        work = {'computed_input_tokens': policy['zero'], 'padded_input_tokens': policy['zero']}
        if cold:
            ids, mask = padded(list(map(list, cold)), self.backend.tokenizer.pad_token_id, self.backend.device, 'left')
            output = self.backend.model.model(input_ids=ids, attention_mask=mask,
                position_ids=(mask.cumsum(-1) - policy['position_step']).clamp_min(policy['zero']), use_cache=True)
            values = split_native_cache(output.past_key_values, list(map(len, cold)))
            scores = self.backend.model.lm_head(output.last_hidden_state[:, policy['last_index']])
            states.update(zip(cold, values, strict=True))
            logits.update((sequence, scores[row:row + policy['row_step']]) for row, sequence in enumerate(cold))
            work['computed_input_tokens'] += sum(map(len, cold))
            work['padded_input_tokens'] += ids.numel()
        if warm:
            terminal = []

            def retain(module, args, kwargs, output):
                descriptor = kwargs['ragged_suffix']
                rows = torch.arange(descriptor.batch_size, device=self.backend.device)
                terminal.append(output.last_hidden_state[rows, descriptor.lengths - policy['position_step']])

            handle = self.backend.model.model.register_forward_hook(retain, with_kwargs=True)
            values = self.extend_states(self.backend, [states[sequence] for sequence, _ in warm],
                [list(sequence[len(prefix):]) for sequence, prefix in warm], work)
            handle.remove()
            hidden, = terminal
            scores = self.backend.model.lm_head(hidden)
            states.update(zip([sequence for sequence, _ in warm], values, strict=True))
            logits.update((sequence, scores[row:row + policy['row_step']]) for row, (sequence, _) in enumerate(warm))
        for sequence in unique:
            self.cache.store(sequence, states[sequence], logits[sequence].clone())
        torch.cuda.synchronize(self.backend.device)
        exact = {sequence for sequence, hit in zip(unique, complete, strict=True) if hit}
        self.records.append({**work, 'logical_input_tokens': sum(map(len, sequences)),
            'unique_input_tokens': sum(map(len, unique)), 'matched_prefix_tokens': sum(map(len, prefixes)),
            'requests': len(sequences), 'unique_requests': len(unique), 'cold_rows': len(cold),
            'warm_rows': len(warm), 'exact_rows': sum(tuple(sequence) in exact for sequence in sequences),
            'cache_bytes': self.cache.bytes, 'elapsed_seconds': time.perf_counter() - started,
            'execution': 'native_cached_prefill_terminal_hidden',
            'row_to_unique': [unique.index(tuple(sequence)) for sequence in sequences]})
        return [states[tuple(sequence)] for sequence in sequences], torch.cat([logits[tuple(sequence)] for sequence in sequences])


class TerminalHistoryDecode(HistoryDecode):
    @torch.inference_mode()
    def prefill(self, inputs):
        self.sequences = self.history.sequences
        policy = self.history.settings
        prefixes = [self.history.cache.prefix(sequence[:policy['exclude_last']]) for sequence in self.sequences]
        complete = [tuple(sequence) in self.history.cache.outputs and self.history.cache.outputs[tuple(sequence)] is not None
                    for sequence in self.sequences]
        if not any(prefixes) and not any(complete):
            return super().prefill(inputs)
        started = time.perf_counter()
        states, logits = self.history.compute_terminal(self.sequences)
        self.logits.copy_(logits)
        chosen = self.logits.argmax(-1)
        load_native_prefixes(self, states, chosen.tolist(), policy['state_copy'])
        self.ids.copy_(chosen[:, None])
        torch.cuda.synchronize(self.backend.device)
        self.history.records[-policy['record_offset']]['elapsed_seconds'] = time.perf_counter() - started
        return self.ids[:, policy['first_index']].clone()


class MeasuredPrefill:
    def prefill(self, inputs):
        forwards = []

        def record(module, args, kwargs):
            value = kwargs['input_ids'] if kwargs.get('input_ids') is not None else kwargs['inputs_embeds']
            forwards.append({'input_shape': list(value.shape[:2]), 'ragged_suffix': kwargs.get('ragged_suffix') is not None})

        handle = self.trunk.register_forward_pre_hook(record, with_kwargs=True)
        torch.cuda.synchronize(self.backend.device)
        started = time.perf_counter()
        value = super().prefill(inputs)
        torch.cuda.synchronize(self.backend.device)
        self.prefill_evidence = {'seconds': time.perf_counter() - started, 'trunk_forwards': forwards}
        handle.remove()
        return value


class NativeDecode(MeasuredPrefill, CapturedDecode):
    pass


class CachedDecode(MeasuredPrefill, TerminalHistoryDecode):
    pass


class MeasuredService:
    def _generate_tokens(self, inputs, options, stopping):
        output = super()._generate_tokens(inputs, options, stopping)
        decoder = next(reversed(self.decoders.values()))
        self.graph_stats['prefill_evidence'] = decoder.prefill_evidence
        return output


class NativeService(MeasuredService, StableGraphFiniteService):
    def make_decoder(self, size, capacity):
        return NativeDecode(self.backend, size, capacity, arena=self.cache_arena,
                            graph_pool=self.graph_pool, graph_stream=self.graph_stream)


class CachedService(MeasuredService, HistoryBatchService):
    def __init__(self, backend, shared, deadlines, *, settings, prompts):
        self.history = TerminalCheckpointHistory(backend, settings['history_cache'])
        BatchService.__init__(self, backend, shared, deadlines, settings=settings, prompts=prompts)
        self.execution_metadata['history_cache'] = settings['history_cache']

    def make_decoder(self, size, capacity):
        decoder = CachedDecode(self.backend, size, capacity, arena=self.cache_arena,
                               graph_pool=self.graph_pool, graph_stream=self.graph_stream)
        decoder.history = self.history
        return decoder


def allow_exact_cache_without_trunk_call():
    tree = ast.parse(textwrap.dedent(inspect.getsource(BatchService._generate)))
    function, = tree.body
    check, = [node for node in function.body if isinstance(node, ast.Assert)
              and ast.unparse(node.test) == 'shapes and len(texts) == len(batch)']
    check.test = ast.parse("(shapes or self.graph_stats['history_prefill']['exact_rows'] == len(batch)) and len(texts) == len(batch)", mode='eval').body
    for node in ast.walk(function):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'super' and not node.args:
            node.args = [ast.Name(id='BatchService', ctx=ast.Load()), ast.Name(id='self', ctx=ast.Load())]
    ast.fix_missing_locations(tree)
    namespace = dict(BatchService._generate.__globals__, BatchService=BatchService)
    exec(compile(tree, inspect.getsourcefile(BatchService._generate), 'exec'), namespace)
    return namespace[function.name]


CachedService._generate = allow_exact_cache_without_trunk_call()
