import argparse
import inspect
import json
from pathlib import Path
import time
import traceback
from types import MethodType

import torch

from infra.decode_qualification.run import measure, prompts_from_source, save
from jev_spawn.infra.backend import Backend


def bind_text_forward(model):
    text = model.model.language_model
    assert not text.training and not text.gradient_checkpointing
    assert not getattr(text.config, 'output_hidden_states', False)
    assert not getattr(text.config, 'output_attentions', False)
    assert getattr(text.config, 'return_dict', True) is True
    names = ['use_cache', 'vision_feature_layer', 'vision_feature_select_strategy', 'vision_aspect_ratio', 'is_causal']
    defaults = {name: getattr(text.config, name, None) for name in names}
    assert defaults['use_cache'] is True
    assert all(defaults[name] is None for name in names[1:])
    original = text.forward.__func__
    core = inspect.unwrap(original)
    assert core.__code__.co_filename.endswith('models/qwen3_5/modeling_qwen3_5.py')
    assert core.__name__ == 'forward'

    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values=None, inputs_embeds=None, use_cache=None, **kwargs):
        assert kwargs.pop('return_dict', True) is True
        assert not kwargs.get('output_hidden_states', False)
        assert not kwargs.get('output_attentions', False)
        assert not kwargs.get('debug_io', False)
        assert kwargs.get('is_causal') is None
        assert all(kwargs.get(name) is None for name in names[1:4])
        resolved_cache = defaults['use_cache'] if use_cache is None else use_cache
        return core(self, input_ids=input_ids, attention_mask=attention_mask,
                    position_ids=position_ids, past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds, use_cache=resolved_cache, **kwargs)

    text.forward = MethodType(forward, text)
    return {'class': type(text).__name__, 'defaults': defaults,
            'core_file': core.__code__.co_filename, 'core_line': core.__code__.co_firstlineno,
            'scope': 'Text eval calls only. Resolve unchanged use_cache default. No capture outputs, vision defaults, debug context, or causal override are supported. All tensor computation stays in the original forward.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    rows, system, prompts = prompts_from_source(config)
    native = json.loads(Path(config['native_config']).read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    save(args.output / 'protocol.json', {'config': config, 'native': native,
         'task_ids': [row['task_id'] for row in rows], 'system': system, 'prompts': prompts,
         'qualification': 'Original and instance-bound eager static-cache full8-row outputs must be exactly identical before one fullgraph attempt.'})
    backend = Backend(native)
    originals = {}
    for label in ['original_static', 'bound_static', 'bound_compiled']:
        if label == 'bound_static':
            save(args.output / 'binding.json', bind_text_forward(backend.model))
        arm = 'static_compiled' if label == 'bound_compiled' else 'static_eager'
        for phase in ['warmup', 'measured']:
            print(json.dumps({'arm': label, 'phase': phase, 'status': 'started'}), flush=True)
            started = time.perf_counter()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            before = torch.cuda.memory_allocated()
            try:
                record = measure(backend, config, arm, prompts, system, label == 'bound_compiled' and phase == 'measured')
                record['stage_wall_seconds'] = time.perf_counter() - started
                if label == 'original_static':
                    originals[phase] = record['output_ids']
                else:
                    record['identical_rows_to_original_static'] = sum(a == b for a, b in zip(originals[phase], record['output_ids']))
                save(args.output / f'{label}-{phase}.json', record)
                if label == 'bound_static':
                    assert record['output_ids'] == originals[phase], 'Forward binding changed original eager outputs.'
                print(json.dumps({'arm': label, 'phase': phase, 'seconds': record['seconds'],
                                  'graph_replay': record['cuda_graph_replay_observed']}), flush=True)
            except Exception as error:
                torch.cuda.synchronize()
                save(args.output / f'{label}-{phase}-failure.json', {'status': 'not_qualified',
                     'error': f'{type(error).__name__}: {error}', 'traceback': traceback.format_exc(),
                     'wall_seconds': time.perf_counter() - started, 'allocated_before_bytes': before,
                     'allocated_after_bytes': torch.cuda.memory_allocated(),
                     'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                     'peak_reserved_bytes': torch.cuda.max_memory_reserved()})
                raise


if __name__ == '__main__':
    main()
