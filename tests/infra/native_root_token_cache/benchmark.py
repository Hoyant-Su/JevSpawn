import argparse
import ast
from functools import lru_cache
import json
from pathlib import Path
import statistics
import time
from types import SimpleNamespace

from transformers import AutoTokenizer
import yaml

from baselines.common.persistence import save
from jev_spawn.algo.structured import common_prefix
from jev_spawn.infra.configuration import CORE
from jev_spawn.infra.prompts import load_prompt
from jev_spawn.schema import CONTROLLER, controller_prefix, controller_prompts


def root_prefix_tokens(backend, system, context):
    prefix = controller_prefix(context, '')
    rendered, = backend._render([prefix], system)
    return tuple(backend.tokenizer(rendered, add_special_tokens=False)['input_ids'])


def trace_requests(directory, pattern, backend):
    protocol = json.loads((directory / 'protocol.json').read_text())
    runtime = json.loads((directory / 'session-0000/runtime.json').read_text())['service']
    system = CONTROLLER['system'] + '\n\n' + load_prompt('shared.runtime_contract').format(
        contract=json.dumps(runtime['runtime_contract']))
    CONTROLLER['option_template'] = protocol['prompts']['option_template']
    fields = []
    for path in sorted(directory.glob(pattern)):
        task = json.loads(path.read_text())
        pairs = []
        for turn in task['trace']['rounds']:
            pairs.extend((turn[name + '_request'], turn[name + '_decision'])
                         for name in ('frontier', 'operation') if name + '_decision' in turn)
            if 'parent_computations' in turn:
                pairs.extend(pair for blocks in turn['parent_computations'].values()
                             for block in blocks if 'requests' in block
                             for pair in zip(block['requests'], block['decisions'], strict=True))
        for field, decision in pairs:
            if decision['input_tokens']:
                fields.append((decision['submitted_monotonic'], task['task_id'], field))
    requests = []
    for submitted, task_id, field in sorted(fields, key=lambda row: row[0]):
        prompt, = controller_prompts([field['state']], field['question'], field['options'],
            runtime['candidate_labels'][:len(field['options'])], CONTROLLER['output_instruction'],
            contexts=[field['context']], histories=[field.get('history', '')])
        rendered, = backend._render([prompt], system)
        requests.append({'task_id': task_id, 'field': field,
            'messages': [{'role': 'system', 'content': system}],
            'input_ids': backend.tokenizer(rendered, add_special_tokens=False)['input_ids']})
    return requests


def benchmark(requests, backend, capacity, settings):
    def encode(system, context):
        return root_prefix_tokens(backend, system, context)

    keys = [(next(message['content'] for message in request['messages']
                  if message['role'] == 'system'), request['field']['context']) for request in requests]
    reference = [encode(*key) for key in keys]
    cached = lru_cache(maxsize=capacity)(encode)
    candidate = [cached(*key) for key in keys]
    assert reference == candidate
    root_lengths = []
    saved_root_checks = []
    for request, original, retained in zip(requests, reference, candidate, strict=True):
        length = common_prefix([request['input_ids'], original])
        assert length == common_prefix([request['input_ids'], retained])
        root_lengths.append(length)
        if 'root_tokens' in request:
            assert request['root_tokens'] == request['input_ids'][:length]
            saved_root_checks.append(request['task_id'])
    timings = {'uncached': [], 'cached': []}
    functions = {'uncached': encode, 'cached': cached}
    for _ in range(settings['warmups']):
        for function in functions.values():
            for key in keys:
                function(*key)
    cache_statistics = []
    for _ in range(settings['repetitions']):
        for name, function in functions.items():
            cached.cache_clear()
            started = time.perf_counter()
            for key in keys:
                function(*key)
            timings[name].append(time.perf_counter() - started)
            if name == 'cached':
                cache_statistics.append(cached.cache_info()._asdict())
    median = {name: statistics.median(values) for name, values in timings.items()}
    return {'requests': len(requests), 'tasks': len({request['task_id'] for request in requests}),
        'unique_exact_keys': len(set(keys)), 'all_prefix_tokens_equal': True,
        'all_common_prefix_lengths_equal': True, 'saved_root_token_checks': len(saved_root_checks),
        'root_token_lengths': root_lengths, 'cache_statistics': cache_statistics,
        'elapsed_seconds': timings, 'median_elapsed_seconds': median,
        'speedup': median['uncached'] / median['cached']}


def main(settings):
    shared = yaml.safe_load(Path(settings['shared_config']).read_text())
    tokenizer = AutoTokenizer.from_pretrained(shared['model']['path'], **settings['tokenizer'])
    backend = SimpleNamespace(tokenizer=tokenizer)
    source = ast.parse(Path(settings['backend_source']).read_text())
    definition, = [node for node in source.body if isinstance(node, ast.ClassDef)
                   and node.name == settings['backend_class']]
    render, = [node for node in definition.body if isinstance(node, ast.FunctionDef)
               and node.name == settings['render_method']]
    namespace = {'CORE': CORE}
    exec(compile(ast.Module(body=[render], type_ignores=[]), settings['backend_source'], 'exec'), namespace)
    backend._render = namespace[settings['render_method']].__get__(backend)
    sources = [(str(directory), trace_requests(Path(directory), settings['task_glob'], backend))
               for directory in settings['trace_runs']]
    sources.extend((' + '.join(paths), [request for path in paths
                    for request in json.loads(Path(path).read_text())['requests']])
                   for paths in settings['profile_sources'])
    capacity = shared['runtime']['root_batch_size']
    report = {'scope': 'CPU-only exact root-prefix render/tokenization replay; full prompt tokenization, GPU prefill and model execution are excluded.',
        'config': settings, 'cache_capacity': capacity,
        'key': '(exact system string, exact context string)',
        'trace_reconstruction': 'Stage055 uses actual saved fields in submitted order with current unchanged controller templates and saved runtime contract. Stage059 directly checks saved input_ids and root_tokens.',
        'results': [{'source': name, **benchmark(requests, backend, capacity, settings)}
                    for name, requests in sources]}
    save(settings['output'], report)
    print(json.dumps([{key: value for key, value in row.items()
                      if key not in ('root_token_lengths', 'elapsed_seconds', 'cache_statistics')}
                     for row in report['results']], indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    main(json.loads(parser.parse_args().config.read_text()))
