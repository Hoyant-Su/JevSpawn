import argparse
import json
from pathlib import Path
from statistics import median
from types import SimpleNamespace

import torch
import torch.distributed as dist

from baselines.common.config import SharedConfig
from baselines.common.parallel_run import initialize_parallel
from jev_spawn.infra.readout_labels import AdmittedPrompt
from jev_spawn.infra.stable_finite_graph import StableFiniteGraphTail
from jev_spawn.runtime.prefix_cache import PrefixCache
from tests.infra.native_direct_finite.candidate import DirectFiniteTail
from tests.infra.native_suffix_graph.qualify import compare, measure, tensors


@torch.inference_mode()
def run(settings):
    shared = SharedConfig.load(settings['shared_config'])
    backend, commands, startup = initialize_parallel(
        shared, json.loads(Path(settings['parallel_settings']).read_text()))
    service = json.loads(Path(settings['service']).read_text())['settings']
    destination = Path(settings['output'])
    destination.mkdir(parents=True, exist_ok=True)
    rows = []
    for source in settings['sources']:
        recorded = json.loads(Path(source).read_text())['requests']
        requests = []
        for item in recorded:
            rendered = backend.tokenizer.apply_chat_template(item['messages'], tokenize=False,
                add_generation_prompt=True, enable_thinking=False)
            assert backend.tokenizer(rendered, add_special_tokens=False)['input_ids'] == item['input_ids']
            requests.append(SimpleNamespace(task_id=item['task_id'], field=item['field'],
                admitted=AdmittedPrompt(rendered, tuple(item['input_ids']))))
        roots = [len(item['root_tokens']) for item in recorded]
        root_cache = PrefixCache(shared.runtime.root_batch_size)
        outputs, timings, states, final_states, cold = {}, {}, {}, {}, {}
        for name, constructor in zip(settings['arms'], (StableFiniteGraphTail, DirectFiniteTail), strict=True):
            tail = constructor(backend, shared.runtime, PrefixCache(shared.runtime.root_batch_size),
                               service['state_copy'], service['graph_shape'])

            def operation():
                tail.prefix_cache.clear()
                return tail.score(requests, roots, root_cache)

            result = operation()
            cold[name] = result['elapsed_seconds']
            outputs[name] = {'answers': [answer['choice'] for group in result['groups'] for answer in group],
                'logits': tail.last_logits.clone(), 'computed_input_tokens': result['computed_input_tokens'],
                'root_prefix_tokens': result['root_prefix_tokens'], 'prefix_tokens': result['prefix_tokens'],
                'suffix_tokens': result['suffix_tokens']}
            states[name] = [tensor.clone() for cache in root_cache.entries.values() for tensor in tensors(cache)]
            final_states[name] = [tensor.clone() for cache in tail.prefix_cache.entries.values()
                                 for tensor in tensors(cache)]
            for _ in range(shared.runtime.graph_warmup_steps):
                warm = operation()
            outputs[name]['warm_workload'] = {key: warm[key] for key in (
                'computed_input_tokens', 'padded_input_tokens', 'reused_root_tokens',
                'batch_size', 'physical_batch_size', 'logical_field_count')}
            timings[name] = measure(operation, backend.device, settings['repetitions'])
            tail.graphs.clear()
        left, right = (outputs[name] for name in settings['arms'])
        checks = {'logits': compare([left['logits']], [right['logits']]),
                  'answers_equal': left['answers'] == right['answers'],
                  'final_states': compare(*(final_states[name] for name in settings['arms'])),
                  'root_unchanged': compare(*(states[name] for name in settings['arms']))}
        medians = {name: median(row['wall_seconds'] for row in samples) for name, samples in timings.items()}
        for result in outputs.values():
            result['logits'] = result['logits'].tolist()
        rows.append({'source': source, 'batch_size': len(requests),
            'outputs': outputs, 'checks': checks, 'timings': timings, 'median_wall_seconds': medians,
            'speedup': medians[settings['arms'][0]] / medians[settings['arms'][1]],
            'first_call_seconds': cold,
            'scope': 'Real complete finite calls with warm root caches; final-prefix cache cleared for every repetition. '
                     'First reference call includes root-cache construction while candidate reuses those exact states; '
                     'first-call durations are not a paired speed comparison. Selected probabilities come from the actual model.'})
        (destination / f'rank-{dist.get_rank()}.json').write_text(json.dumps(
            {'settings': settings, 'startup': startup, 'rows': rows}, indent=2) + '\n')
    dist.destroy_process_group(commands.control_group)
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    run(json.loads(parser.parse_args().config.read_text()))
