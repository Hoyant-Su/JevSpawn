import argparse
import json
from pathlib import Path
import statistics
import time

from transformers import AutoTokenizer
import yaml

from jev_spawn.infra.readout_labels import native_labels, validate_boundaries
from jev_spawn.schema import CONTROLLER, controller_prompts


def full_prompt_reference(tokenizer, rendered, sequences, labels, token_ids, counts):
    joined = tokenizer([text + label for text, count in zip(rendered, counts, strict=True)
                        for label in labels[:count]], add_special_tokens=False)['input_ids']
    expected = [sequence + [token] for sequence, count in zip(sequences, counts, strict=True)
                for token in token_ids[:count]]
    assert joined == expected


def run(settings):
    shared = yaml.safe_load(Path(settings['shared_config']).read_text())
    tokenizer = AutoTokenizer.from_pretrained(shared['model']['path'], local_files_only=True)
    labels, token_ids = native_labels(tokenizer, json.loads(Path(settings['label_config']).read_text()))
    real = json.loads(Path(settings['source_inputs']).read_text())
    fixture = json.loads(Path(settings['fixture']).read_text())
    options = [{'id': fixture['option_id'].format(index=index),
                'description': fixture['description'].format(index=index)} for index in range(len(labels))]
    prompts = controller_prompts(fixture['states'], fixture['question'], options, labels, CONTROLLER['output_instruction'])
    rendered = tokenizer.apply_chat_template(
        [[{'role': 'system', 'content': CONTROLLER['system']}, {'role': 'user', 'content': prompt}]
         for prompt in prompts], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    workloads = [
        ('recorded_fields', real['rendered'], real['input_ids'], [len(field['options']) for field in real['fields']]),
        ('complete_native_label_pool', rendered, tokenizer(rendered, add_special_tokens=False)['input_ids'],
         [len(labels)] * len(rendered)),
    ]
    rows = []
    for name, texts, sequences, counts in workloads:
        arguments = (tokenizer, texts, sequences, labels, token_ids, counts)
        full_prompt_reference(*arguments)
        validate_boundaries(*arguments)
        samples = {'full_prompt': [], 'atomic_suffix': []}
        for _ in range(settings['repeats']):
            for label, function in [('full_prompt', full_prompt_reference), ('atomic_suffix', validate_boundaries)]:
                start = time.perf_counter()
                function(*arguments)
                samples[label].append(time.perf_counter() - start)
        rows.append({'name': name, 'input_lengths': list(map(len, sequences)), 'candidate_counts': counts,
                     'same_boundary_result': True, 'seconds': samples,
                     'median_seconds': {name: statistics.median(times) for name, times in samples.items()}})
    result = {'settings': settings, 'workloads': rows}
    Path(settings['output']).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.config.read_text()))
