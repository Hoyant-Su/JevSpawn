import argparse
from copy import deepcopy
import json
from pathlib import Path
import time
from types import SimpleNamespace

from baselines.common.resources import TEMPLATES
from jev_spawn.infra.prompts import load_prompt
from jev_spawn.infra.readout_labels import AdmittedPrompt
from jev_spawn.runtime.reference_transport import unpack_references
from jev_spawn.runtime.shared_reference_state import relocate, restore
from jev_spawn.schema import CONTROLLER, controller_prompts
from replay_recorded_finite import main, prepare


def prepare_relocated(config, backend, shared):
    cohorts, original_seconds = prepare(config, backend, shared)
    started = time.perf_counter()
    layout = json.loads(Path(config['layout_config']).read_text())
    transport = json.loads(Path(layout['transport']).read_text())
    prompt = load_prompt(layout['prompt'])['encoded_state']
    expected = json.loads(Path(layout['rewritten_requests']).read_text())
    relocated = []
    for (requests, lengths), expected_rows in zip(cohorts[config['original_variant']], expected, strict=True):
        rows = []
        for request, recorded in zip(requests, expected_rows, strict=True):
            field = deepcopy(request.field)
            segments = relocate(field['input'], transport, layout['shared_keys'])
            original = unpack_references(field['input'], transport)
            assert json.dumps(original, **layout['canonical_serialization']) == json.dumps(restore(segments, transport), **layout['canonical_serialization'])
            state = prompt.format(**{name: json.dumps(value, **layout['serialization']) for name, value in segments.items()})
            field['state'] = TEMPLATES[layout['state_template_key']].format(context=field['context'], state=state)
            user, = controller_prompts([field['state']], field['question'], field['options'],
                backend.answer_labels[:len(field['options'])], CONTROLLER['output_instruction'])
            messages = [{'role': 'system', 'content': CONTROLLER['system']}, {'role': 'user', 'content': user}]
            rendered = backend.tokenizer.apply_chat_template(messages, tokenize=False,
                add_generation_prompt=layout['tokenizer']['add_generation_prompt'], enable_thinking=shared.generation.enable_thinking)
            tokens = backend.tokenizer(rendered, add_special_tokens=layout['tokenizer']['add_special_tokens'])['input_ids']
            assert request.task_id == recorded['task_id'] and field['id'] == recorded['field']['id']
            assert rendered == recorded['rendered'] and tokens == recorded['tokens']
            assert len(tokens) <= shared.model.max_input_tokens
            rows.append(SimpleNamespace(task_id=request.task_id, field=field, input_ids=tokens,
                                        admitted=AdmittedPrompt(rendered, tuple(tokens))))
        assert all(row.input_ids[:length] == original.input_ids[:length]
                   for row, original, length in zip(rows, requests, lengths, strict=True))
        relocated.append((rows, lengths))
    cohorts[config['relocated_variant']] = relocated
    return cohorts, original_seconds + time.perf_counter() - started


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    main(json.loads(parser.parse_args().config.read_text()), prepare_relocated)
