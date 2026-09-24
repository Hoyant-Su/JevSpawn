import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist

from baselines.common.config import SharedConfig
from baselines.common.parallel_run import initialize_parallel
from jev_spawn.infra.history_cache import HistoryTail
from jev_spawn.infra.readout_labels import AdmittedPrompt
from jev_spawn.infra.stable_finite_graph import StableFiniteGraphTail
from jev_spawn.runtime.prefix_cache import PrefixCache
from tests.infra.native_suffix_graph.qualify import compare, measure, tensors


def reset_history(tail):
    tail.history.entries.clear()


def requests_from_record(backend, source):
    recorded = json.loads(Path(source).read_text())['requests']
    requests = []
    for item in recorded:
        rendered = backend.tokenizer.apply_chat_template(item['messages'], tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
        assert backend.tokenizer(rendered, add_special_tokens=False)['input_ids'] == item['input_ids']
        requests.append(SimpleNamespace(task_id=item['task_id'], field=item['field'],
            messages=item['messages'], admitted=AdmittedPrompt(rendered, tuple(item['input_ids']))))
    return requests, [len(item['root_tokens']) for item in recorded]


@torch.inference_mode()
def run(settings):
    shared = SharedConfig.load(settings['shared_config'])
    backend, commands, startup = initialize_parallel(
        shared, json.loads(Path(settings['parallel_settings']).read_text()))
    service = json.loads(Path(settings['service']).read_text())['settings']
    output = Path(settings['output'])
    output.mkdir(parents=True, exist_ok=True)
    report = {'settings': settings, 'startup': startup, 'sequences': []}
    for sources in settings['sequences']:
        sequence = [requests_from_record(backend, source) for source in sources]
        roots = PrefixCache(shared.runtime.root_batch_size)
        reference = StableFiniteGraphTail(backend, shared.runtime,
            PrefixCache(shared.runtime.root_batch_size), service['state_copy'], service['graph_shape'])
        candidate = HistoryTail(backend, shared.runtime, PrefixCache(shared.runtime.root_batch_size),
            service['state_copy'], service['graph_shape'], settings['history_cache'])
        tails = dict(zip(settings['arms'], (reference, candidate), strict=True))
        diagnostics = []

        def reset():
            for tail in tails.values():
                tail.prefix_cache.clear()
            reset_history(candidate)

        # Warm every actual root before paired timing; neither arm receives a cold-root advantage.
        for requests, lengths in sequence:
            reference.score(requests, lengths, roots)
        reset()
        for source, (requests, lengths) in zip(sources, sequence, strict=True):
            values, logits = {}, {}
            for name, tail in tails.items():
                result = tail.score(requests, lengths, roots)
                values[name] = {key: result[key] for key in settings['workload_fields']}
                values[name]['choices'] = [answer['choice'] for group in result['groups'] for answer in group]
                logits[name] = tail.last_logits.clone()
            keys = tuple(reference.prefix_cache.entries)
            assert keys == tuple(candidate.prefix_cache.entries)
            diagnostics.append({'source': source, 'task_ids': [r.task_id for r in requests],
                'workload': values, 'logits': compare(list(logits.values())[:1], list(logits.values())[1:]),
                'choices_equal': values[settings['arms'][0]]['choices'] == values[settings['arms'][1]]['choices'],
                'native_states': compare(
                    [t for key in keys for t in tensors(reference.prefix_cache.entries[key])],
                    [t for key in keys for t in tensors(candidate.prefix_cache.entries[key])])})
        timings = []
        for _ in range(settings['repetitions']):
            reset()
            cohorts = []
            for requests, lengths in sequence:
                cohorts.append({name: measure(lambda: tail.score(requests, lengths, roots),
                    backend.device, settings['measurements_per_cohort'])
                    for name, tail in tails.items()})
            timings.append(cohorts)
        report['sequences'].append({'sources': sources, 'diagnostics': diagnostics, 'timings': timings})
        (output / settings['rank_file'].format(rank=dist.get_rank())).write_text(
            json.dumps(report, indent=2) + '\n')
        for tail in tails.values():
            tail.graphs.clear()
    dist.destroy_process_group(commands.control_group)
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    run(json.loads(parser.parse_args().config.read_text()))
