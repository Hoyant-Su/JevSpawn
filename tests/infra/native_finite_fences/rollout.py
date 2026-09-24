import argparse
import json
from pathlib import Path

from baselines.common.parallel_run import execute
from candidate import without_timing_fences
from jev_spawn.infra import cached_suffix, finite_batch, finite_graph


def run(settings):
    specification = json.loads(Path(settings['specification']).read_text())
    assert settings['shared_config'] == specification['shared_config']
    if settings['remove_timing_fences']:
        patch = settings['patch']
        finite_graph.score_finite_with_tail = without_timing_fences(
            finite_batch.score_finite_with_tail, patch['score_fences'])
        finite_graph.FiniteGraphTail.__call__ = without_timing_fences(
            finite_graph.FiniteGraphTail.__call__, patch['tail_fences'])
        finite_graph.RaggedFiniteGraphTail.extend_states = staticmethod(without_timing_fences(
            cached_suffix.ragged_suffix, patch['suffix_fences']))
        score = finite_graph.FiniteGraphTail.score

        def observed(tail, *args, **kwargs):
            result = score(tail, *args, **kwargs)
            result['unfenced_host_timings'] = result.pop('timings')
            result['timings'] = {'capture_seconds': result['unfenced_host_timings']['capture_seconds']}
            for name in settings['work_timing_fields']:
                result['unfenced_host_timings'][name] = result.pop(name)
            result['component_timing_scope'] = settings['component_timing_scope']
            return result

        finite_graph.FiniteGraphTail.score = observed
    execute(specification, Path(settings['run_output']),
            json.loads(Path(settings['parallel_settings']).read_text()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    run(json.loads(parser.parse_args().config.read_text()))
