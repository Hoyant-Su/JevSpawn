import argparse
import json
from pathlib import Path
from statistics import median
from types import SimpleNamespace

import torch
import torch.distributed as dist

from baselines.common.config import SharedConfig
from baselines.common.parallel_run import initialize_parallel
from candidate import without_timing_fences
from jev_spawn.infra import cached_suffix, finite_batch, finite_graph
from jev_spawn.infra.readout_labels import AdmittedPrompt
from jev_spawn.infra.stable_finite_graph import StableFiniteGraphTail
from jev_spawn.runtime.prefix_cache import PrefixCache
from tests.infra.native_suffix_graph.qualify import compare, measure, tensors


@torch.inference_mode()
def run(settings):
    shared = SharedConfig.load(settings['shared_config'])
    backend, commands, startup = initialize_parallel(
        shared, json.loads(Path(settings['parallel_settings']).read_text()))
    service = json.loads(Path(settings['service']).read_text())['settings']
    patches = settings['patch']
    reference = (finite_batch.score_finite_with_tail, finite_graph.FiniteGraphTail.__call__,
                 cached_suffix.ragged_suffix)
    candidate = tuple(without_timing_fences(function, patches[key]) for function, key in
                      zip(reference, settings['patch_order'], strict=True))
    destination = Path(settings['output'])
    destination.mkdir(parents=True, exist_ok=True)
    rows = []
    for source in settings['sources']:
        recorded = json.loads(Path(source).read_text())['requests']
        requests = []
        for item in recorded:
            rendered = backend.tokenizer.apply_chat_template(
                item['messages'], tokenize=False, add_generation_prompt=True, enable_thinking=False)
            assert backend.tokenizer(rendered, add_special_tokens=False)['input_ids'] == item['input_ids']
            requests.append(SimpleNamespace(task_id=item['task_id'], field=item['field'],
                admitted=AdmittedPrompt(rendered, tuple(item['input_ids']))))
        roots = [len(item['root_tokens']) for item in recorded]
        root_cache = PrefixCache(shared.runtime.root_batch_size)
        tail = StableFiniteGraphTail(backend, shared.runtime,
            PrefixCache(shared.runtime.root_batch_size), service['state_copy'], service['graph_shape'])

        def operation():
            tail.prefix_cache.clear()
            return tail.score(requests, roots, root_cache)

        outputs, timing = {}, {}
        for name, implementation in zip(settings['arms'], (reference, candidate), strict=True):
            finite_graph.score_finite_with_tail, finite_graph.FiniteGraphTail.__call__, extension = implementation
            finite_graph.RaggedFiniteGraphTail.extend_states = staticmethod(extension)
            result = operation()
            outputs[name] = {
                'answers': result['groups'], 'logits': tail.last_logits.clone(),
                'root': [value.clone() for cache in root_cache.entries.values() for value in tensors(cache)],
                'suffix': [value.clone() for cache in tail.prefix_cache.entries.values() for value in tensors(cache)]}
            for _ in range(shared.runtime.graph_warmup_steps):
                operation()
            timing[name] = measure(operation, backend.device, settings['repetitions'])
        left, right = (outputs[name] for name in settings['arms'])
        checks = {key: compare(left[key], right[key]) for key in ('root', 'suffix')}
        checks['logits'] = compare([left['logits']], [right['logits']])
        checks['answers_equal'] = left['answers'] == right['answers']
        rows.append({'source': source, 'batch_size': len(requests),
            'checks': checks, 'timings': timing,
            'median_wall_seconds': {name: median(row['wall_seconds'] for row in samples)
                                    for name, samples in timing.items()}})
        (destination / f'rank-{dist.get_rank()}.json').write_text(json.dumps(
            {'settings': settings, 'startup': startup, 'rows': rows}, indent=2) + '\n')
        assert checks['answers_equal'] and all(checks[key]['equal'] for key in ('root', 'suffix', 'logits'))
        finite_graph.score_finite_with_tail, finite_graph.FiniteGraphTail.__call__, extension = reference
        finite_graph.RaggedFiniteGraphTail.extend_states = staticmethod(extension)
        tail.graphs.clear()
    dist.destroy_process_group(commands.control_group)
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    run(json.loads(parser.parse_args().config.read_text()))
