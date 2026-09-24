import argparse
import json
from pathlib import Path


def summarize(config, destination):
    source = json.loads(Path(config['result']).read_text())
    records = source['records']
    paired = []
    for phase in config['phases']:
        by_variant = {record['variant']: record for record in records if record['phase'] == phase['name']}
        for variants in config['comparison_pairs']:
            reference, candidate = (by_variant[name] for name in variants)
            assert reference['task_ids'] == candidate['task_ids']
            rows = []
            for left, right in zip(reference['answers'], candidate['answers'], strict=True):
                assert left['id'] == right['id'] and left['option_ids'] == right['option_ids']
                assert left['input_tokens'] == right['input_tokens']
                rows.append({'node': left['id'], 'choice_equal': left['choice'] == right['choice'],
                             'max_logit_difference': max(abs(a - b) for a, b in
                                 zip(left['option_logits'], right['option_logits'], strict=True))})
            paired.append({'phase': phase['name'], 'variants': variants, 'reference_seconds': reference['rankmax_seconds'],
                           'candidate_seconds': candidate['rankmax_seconds'],
                           'speedup': reference['rankmax_seconds'] / candidate['rankmax_seconds'],
                           'decisions_per_second': len(rows) / candidate['rankmax_seconds'],
                           'choices_equal': all(row['choice_equal'] for row in rows), 'rows': rows,
                           'candidate_timings': candidate['details'][0]['timings'],
                           'candidate_peak_cuda_memory_bytes': candidate['details'][0]['peak_cuda_memory_bytes']})
    rank_parity = []
    for rank in source['startup']:
        for record in records:
            path = Path(config['output'], config['rank_file'].format(
                rank=rank['rank'], phase=record['phase'], variant=record['variant']))
            saved = json.loads(path.read_text())
            rank_parity.append({'rank': rank['rank'], 'phase': record['phase'], 'variant': record['variant'],
                                'answers_equal': saved['answers'] == record['answers']})
    result = {'source': config['result'], 'paired': paired, 'ranks': rank_parity,
              'timing_scope': 'Finite calls including exact prefix computation and state loading; not complete task latency or autoregressive token throughput.'}
    Path(destination).write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    summarize(json.loads(args.config.read_text()), args.output)
