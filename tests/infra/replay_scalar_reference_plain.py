import argparse
from collections import defaultdict
import json
from pathlib import Path
import time
from types import SimpleNamespace

from baselines.common.task_prefix_service import TaskPrefixService
from jev_spawn.infra.readout_labels import AdmittedPrompt, validate_boundaries
from jev_spawn.schema import CONTROLLER, controller_prompts
from replay_recorded_finite import main


def prepare(config, backend, shared):
    started = time.perf_counter()
    pairs = json.loads(Path(config['paired_inputs']).read_text())
    protocol = json.loads((Path(config['source_run']) / 'protocol.json').read_text())
    CONTROLLER['option_template'] = protocol['prompts']['option_template']
    assert len(pairs) == config['expected_requests']
    cohorts = {}
    for name in config['variants']:
        grouped = defaultdict(list)
        for pair in pairs:
            saved = pair['variants'][name]
            field = saved['field']
            count = len(field['options'])
            assert backend.answer_labels[:count] == pair['labels']
            assert backend.answer_label_ids[:count] == pair['label_ids']
            user, = controller_prompts([field['state']], field['question'], field['options'],
                                      pair['labels'], CONTROLLER['output_instruction'])
            messages = [{'role': 'system', 'content': CONTROLLER['system']}, {'role': 'user', 'content': user}]
            rendered = backend.tokenizer.apply_chat_template(messages, tokenize=False,
                add_generation_prompt=True, enable_thinking=shared.generation.enable_thinking)
            tokens = backend.tokenizer(rendered, add_special_tokens=False)['input_ids']
            assert messages == saved['messages'] and rendered == saved['rendered'] and tokens == saved['tokens']
            assert len(tokens) == saved['input_tokens'] <= shared.model.max_input_tokens
            validate_boundaries(backend.tokenizer, [rendered], [tokens], backend.answer_labels,
                                backend.answer_label_ids, [count])
            grouped[pair['source_cohort']].append(SimpleNamespace(task_id=pair['task_id'], field=field,
                input_ids=tokens, admitted=AdmittedPrompt(rendered, tuple(tokens))))
        cohorts[name] = [(requests, [TaskPrefixService.task_prefix_length(
            SimpleNamespace(backend=backend), [request]) for request in requests])
            for requests in grouped.values()]
        assert sum(len(requests) for requests, _ in cohorts[name]) == config['expected_requests']
    return cohorts, time.perf_counter() - started


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    main(json.loads(parser.parse_args().config.read_text()), prepare)
