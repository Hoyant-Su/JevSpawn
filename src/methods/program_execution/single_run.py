import argparse
import json
import statistics
import string
from pathlib import Path

from transformers import AutoTokenizer

from jev_spawn.infra.backend import Backend
from jev_spawn.schema import CONTROLLER, controller_prompts
from methods.decision_program.compiler import compile_programs, instantiate
from methods.decision_program.run import execute
from methods.generated_schema.run import write
from methods.program_execution.tiled_run import direct, evaluate_rankings, flattened, jsonl, memory, parity, read
from jev_spawn.infra.prompts import load_prompt, resolve_prompts


def preflight(config, jobs, collections, programs, settings):
    backend = Backend.__new__(Backend)
    backend.tokenizer = AutoTokenizer.from_pretrained(config['model_path'], local_files_only=True)
    prompts = load_prompt(settings['direct_prompt_schema'])
    records = []
    for job, root in zip(jobs, collections):
        generated = instantiate(job, programs[job['task_id']])
        direct_nodes = [{'state': json.dumps({'query': root['query'], 'document_id': doc['document_id'],
                         'document': doc['text'], 'worker_messages': []}, ensure_ascii=False),
                         'question': prompts['relevance_question'], 'options': prompts['relevance_options']}
                        for doc in root['candidates']]
        for method, nodes in [('generated', generated), ('direct', direct_nodes)]:
            texts = [controller_prompts([node['state']], node['question'], node['options'],
                     list(string.ascii_uppercase[:len(node['options'])]), CONTROLLER['output_instruction'])[0]
                     for node in nodes]
            lengths = list(map(len, backend.tokenizer(backend._render(texts, CONTROLLER['system']),
                                                      add_special_tokens=False)['input_ids']))
            assert max(lengths) <= config['max_input_tokens']
            records.append({'task_id': job['task_id'], 'method': method, 'workers': len(nodes),
                            'minimum_tokens': min(lengths), 'maximum_tokens': max(lengths),
                            'total_tokens': sum(lengths), 'truncated': False})
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--settings', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    settings = resolve_prompts(read(args.settings))
    config = resolve_prompts(read(settings['native_config']))
    jobs, collections = jsonl(settings['jobs']), jsonl(settings['collections'])
    frozen = read(settings['frozen_compilation'])['programs']
    assert settings['task_execution'] == 'sequential_single_root'
    assert config['batch_size'] == settings['batch_size'] == 1
    assert config['branch_batch_size'] == settings['live_leaf_cap'] == 128
    assert len(jobs) == len(collections) == settings['job_count'] == 8
    assert settings['modes'] == settings['direct_modes'] == ['tiled_independent', 'tiled_shared']
    for job, root in zip(jobs, collections):
        assert job['task_id'] == root['task_id']
        assert len(job['items']) == len({item['id'] for item in job['items']}) == settings['items_per_root']
        assert [(item['id'], item['input']['document']) for item in job['items']] == [
            (doc['document_id'], doc['text']) for doc in root['candidates']]
    CONTROLLER['option_template'] = settings['option_template']
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output / 'protocol.json', {'settings': settings, 'native': config})
    write(args.output / 'preflight.json', preflight(config, jobs, collections, frozen, settings))
    if args.preflight_only:
        return
    backend = Backend(config)
    write(args.output / 'model-load-memory.json', memory(backend))
    write(args.output / 'backend.json', backend.metadata)
    for phase in ('warmup', 'measured'):
        phase_dir = args.output / phase
        phase_dir.mkdir()
        summaries = []
        for index, (job, root) in enumerate(zip(jobs, collections)):
            directory = phase_dir / f'root-{index:02d}'
            directory.mkdir()
            before = memory(backend)
            compiled = compile_programs(backend, [job], settings)
            write(directory / 'compilation.json', compiled)
            write(directory / 'compiler-memory.json', {'before': before, 'after': memory(backend),
                  'peak_allocated_bytes': compiled['generation']['peak_allocated_bytes'],
                  'peak_reserved_bytes': compiled['generation']['peak_reserved_bytes']})
            assert not compiled['failures'] and compiled['programs'] == {job['task_id']: frozen[job['task_id']]}
            summary = execute(backend, [job], compiled['programs'], settings, directory)
            compiler_seconds = compiled['generation']['elapsed_seconds']
            for value in summary.values():
                value.update(compiler_seconds=compiler_seconds, total_seconds=compiler_seconds + value['execution_seconds'])
            summary['direct'] = direct(backend, [root], settings, directory)
            row = {'task_id': job['task_id'], 'directory': str(directory), 'summary': summary}
            summaries.append(row)
            write(phase_dir / 'summary.json', summaries)
            print(json.dumps({'phase': phase, **row}), flush=True)
    gold = {row['task_id']: row['relevant_document_ids'] for row in jsonl(settings['relevance'])}
    evaluation = []
    for index, root in enumerate(collections):
        directory = args.output / 'measured' / f'root-{index:02d}'
        independent = flattened(read(directory / 'tiled_independent-calls.json'))
        metrics = {}
        for prefix in ('', 'direct-'):
            for mode in settings['modes']:
                name = prefix + mode
                metrics[name] = evaluate_rankings([root], gold, read(directory / f'{name}-outputs.json'), settings['ranking_cutoff'])
                decisions = flattened(read(directory / f'{name}-calls.json'))
                metrics[name]['warm_measured'] = parity(decisions, flattened(read(
                    args.output / 'warmup' / f'root-{index:02d}' / f'{name}-calls.json')))
                if prefix == '':
                    metrics[name]['versus_tiled_independent'] = parity(decisions, independent)
        evaluation.append({'task_id': root['task_id'], 'metrics': metrics})
    write(args.output / 'evaluation.json', {'queries': evaluation, 'macro_ndcg': {
        name: statistics.mean(row['metrics'][name]['macro_ndcg'] for row in evaluation)
        for name in evaluation[0]['metrics']}})


if __name__ == '__main__':
    main()
