import argparse
import json
from pathlib import Path
import time

import torch
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNorm, Qwen3_5RMSNormGated

from baselines.common.config import SharedConfig
from tests.baselines.common.profile_decode import measure
from baselines.common.tasks import read, rows, render
from jev_spawn.infra.backend import Backend
from jev_spawn.infra.fused_norm import install, restore, rms_norm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--shared-config', type=Path, required=True)
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config, profile = SharedConfig.load(args.shared_config), read(args.profile)
    args.output.mkdir(parents=True, exist_ok=False)
    backend = Backend(config.backend())
    tasks = rows(profile['tasks'])[profile['offset']:profile['offset'] + profile['samples']]
    texts = backend.tokenizer.apply_chat_template(
        [[{'role': 'user', 'content': render(task)}] for task in tasks],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    comparisons, measurements = [], []

    def compare(module, inputs, output):
        is_gated = type(module) is Qwen3_5RMSNormGated
        actual = rms_norm(inputs[0], module.weight,
                          module.variance_epsilon if is_gated else module.eps,
                          inputs[1] if is_gated else None)
        difference = (output.float() - actual.float()).abs()
        comparisons.append({'kind': type(module).__name__, 'shape': list(output.shape),
                            'max_absolute_error': difference.max().item(),
                            'rms_error': difference.square().mean().sqrt().item(),
                            'rms_reference': output.float().square().mean().sqrt().item(),
                            'equal_fraction': (output == actual).float().mean().item()})
        torch.testing.assert_close(actual, output, rtol=profile['rtol'], atol=profile['atol'])

    try:
        for size in profile['batch_sizes']:
            inputs, _ = backend._encode(texts[:size])
            hooks = [module.register_forward_hook(compare) for module in backend.model.modules()
                     if type(module) in (Qwen3_5RMSNorm, Qwen3_5RMSNormGated)]
            try:
                with torch.inference_mode():
                    backend.model(**inputs, use_cache=False, logits_to_keep=1)
            finally:
                for hook in hooks:
                    hook.remove()
            measure(backend, inputs, 'uninstrumented', profile['tokens'], None)
            for repeat in range(profile['repeats']):
                eager = measure(backend, inputs, 'events', profile['tokens'], None)
                originals = install(backend.model)
                try:
                    measure(backend, inputs, 'uninstrumented', profile['tokens'], None)
                    fused = measure(backend, inputs, 'events', profile['tokens'], None)
                finally:
                    restore(originals)
                measurement = {'batch_size': size, 'repeat': repeat, 'eager': eager, 'fused': fused,
                               'exact_token_parity': eager['output_token_ids'] == fused['output_token_ids']}
                measurements.append(measurement)
                (args.output / 'measurements.json').write_text(json.dumps(measurements, indent=2) + '\n')
                print(json.dumps({'batch_size': size, 'repeat': repeat,
                                  'eager_seconds': eager['elapsed_seconds'], 'fused_seconds': fused['elapsed_seconds'],
                                  'eager_itl_ms': eager['itl_median_ms'], 'fused_itl_ms': fused['itl_median_ms'],
                                  'exact_token_parity': measurement['exact_token_parity']}), flush=True)
            originals = install(backend.model)
            try:
                trace = measure(backend, inputs, 'events', profile['trace_tokens'],
                                args.output / f'fusion-batch-{size}.json')
            finally:
                restore(originals)
            (args.output / f'trace-summary-{size}.json').write_text(json.dumps(trace, indent=2) + '\n')
    finally:
        (args.output / 'operator-comparisons.json').write_text(json.dumps(comparisons, indent=2) + '\n')
    result = {'config': profile, 'model': backend.metadata, 'measurements': measurements,
              'operator_comparisons': len(comparisons),
              'exact_token_parity': all(row['exact_token_parity'] for row in measurements)}
    (args.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
