import argparse
import json
from pathlib import Path
import statistics
import time

import torch

from baselines.common.config import SharedConfig
from jev_spawn.algo import structured
from jev_spawn.infra.backend import Backend
from jev_spawn.infra.prompts import load_prompt
from jev_spawn.infra.readout_labels import validate_boundaries
from jev_spawn.runtime.prefix_cache import PrefixCache
from jev_spawn.schema import CONTROLLER, controller_prompts
from methods.program_execution.grouped import score_grouped
from tests.infra.readout_labels.profile_boundaries import full_prompt_reference


@torch.inference_mode()
def run(settings):
    shared = SharedConfig.load(settings['shared_config'])
    source = json.loads(Path(settings['source_inputs']).read_text())
    method = json.loads(Path(settings['method']).read_text())
    CONTROLLER['option_template'] = load_prompt(method['prompts'])['option_template']
    backend = Backend(shared.backend())
    fields = source['fields']
    prompts = [controller_prompts([field['state']], field['question'], field['options'],
                                  backend.answer_labels[:len(field['options'])], CONTROLLER['output_instruction'])[0]
               for field in fields]
    assert backend._render(prompts, CONTROLLER['system']) == source['rendered']
    cache = PrefixCache(shared.runtime.root_batch_size)
    implementations = {'full_prompt': full_prompt_reference, 'atomic_suffix': validate_boundaries}
    arguments = {'prefix_cache': cache, 'prefix_lengths': [source['prefix_length']]}
    score_grouped(backend, [fields], settings['field_mode'], **arguments)
    rows = []
    for repeat in range(settings['repeats']):
        pair = {}
        for name in settings['orders'][repeat % len(settings['orders'])]:
            structured.validate_boundaries = implementations[name]
            torch.cuda.synchronize(backend.device)
            start = time.perf_counter()
            result = score_grouped(backend, [fields], settings['field_mode'], **arguments)
            torch.cuda.synchronize(backend.device)
            pair[name] = {'seconds': time.perf_counter() - start, 'timings': result['timings'],
                          'computed_input_tokens': result['computed_input_tokens'],
                          'prefix_hit': result['persistent_prefix_hit'], 'fields': result['fields']}
        assert pair['full_prompt']['fields'] == pair['atomic_suffix']['fields']
        assert pair['full_prompt']['computed_input_tokens'] == pair['atomic_suffix']['computed_input_tokens']
        rows.append(pair)
    structured.validate_boundaries = validate_boundaries
    result = {'settings': settings, 'field_count': len(fields), 'prefix_tokens': source['prefix_length'],
              'median_seconds': {name: statistics.median(pair[name]['seconds'] for pair in rows)
                                 for name in implementations}, 'all_finite_outputs_exact': True, 'pairs': rows}
    Path(settings['output']).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: value for key, value in result.items() if key != 'pairs'}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.config.read_text()))
