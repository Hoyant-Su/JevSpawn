import argparse
import copy
import json
import os
from pathlib import Path

from baselines.official.model_service import GenerationService
from baselines.official.phases import run_phase, write
from data.evaluate_bright import ranking_metrics
from jev_spawn.infra.backend import Backend


def main(service_class=GenerationService):
    parser = argparse.ArgumentParser()
    parser.add_argument('--settings', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    settings = json.loads(args.settings.read_text())
    native = json.loads(Path(settings['native_config']).read_text())
    prompts = json.loads(Path(settings['prompts']).read_text())
    rows = [json.loads(line) for line in Path(settings['collections']).read_text().splitlines()]
    formal_ids = {json.loads(line)['task_id']
                  for line in Path(settings['formal_collections']).read_text().splitlines()}
    assert len(rows) == settings['task_count'] == settings['batch_size'] == native['batch_size']
    assert not formal_ids & {row['task_id'] for row in rows}
    assert all(len(row['candidates']) == settings['candidate_count'] for row in rows)
    assert settings['context_length'] == native['max_input_tokens']
    os.environ['EVALTASK'] = 'bright_pony'
    args.output.mkdir(parents=True)
    write(args.output / 'protocol.json', {'settings': settings, 'native': native,
                                        'task_ids': [r['task_id'] for r in rows],
                                        'cuda_visible_devices': os.environ['CUDA_VISIBLE_DEVICES']})
    backend = Backend(native)
    write(args.output / 'backend.json', backend.metadata)
    service = service_class(backend, settings['batch_size'], settings['batch_wait_seconds'])
    service.agent_tokenizers = {row['task_id']: copy.deepcopy(backend.tokenizer) for row in rows}
    try:
        for phase in ['warmup', 'measured']:
            results = run_phase(service, rows, settings, prompts, args.output, phase)
    finally:
        service.close()
    labels = {row['task_id']: row['relevant_document_ids'] for row in
              map(json.loads, Path(settings['relevance']).read_text().splitlines())}
    metrics = []
    for row, result in zip(rows, results):
        ranking = result.get('ranked_document_ids')
        valid = result['status'] == 'completed' and ranking is not None
        value = ranking_metrics(ranking, labels[row['task_id']], settings['ranking_cutoff']) if valid else {'ndcg': 0, 'recall': 0}
        metrics.append({'task_id': row['task_id'], 'valid': valid, **value})
    write(args.output / 'development_quality.json', {
        'queries': len(rows), 'valid': sum(row['valid'] for row in metrics),
        'ndcg': sum(row['ndcg'] for row in metrics) / len(rows),
        'recall': sum(row['recall'] for row in metrics) / len(rows), 'per_query': metrics})


if __name__ == '__main__':
    main()
