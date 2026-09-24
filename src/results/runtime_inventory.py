import csv
import json
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(path.read_text())


def intervals(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if key == 'inter_token_seconds':
                yield from item
            else:
                yield from intervals(item)
    elif isinstance(value, list):
        for item in value:
            yield from intervals(item)


def main():
    observations = []
    runs = sorted(ROOT.glob('runs/official-*/evaluation.json'))
    runs += [ROOT / 'runs' / name / 'evaluation.json' for name in (
        'official-dyflow-supergpqa256-001', 'official-foldagent-bright104-002')]
    runs += sorted(ROOT.glob('runs/single-reasoning-*/evaluation.json'))
    for source in runs:
        directory = source.parent
        blocks = sorted(directory.glob('block-*/complete.json'))
        if not blocks:
            continue
        times, tokens, identities = [], [], []
        seconds = 0
        model_calls = 0
        for path in blocks:
            block = read(path)
            identities.extend(block['task_ids'])
            seconds += block.get('whole_block_seconds', block['metrics']['elapsed_seconds'])
            model_calls += block['metrics']['model_calls']
            if 'results' in block:
                rows = block['results']
                if 'answers_available_monotonic' in block:
                    times.extend([block['answers_available_monotonic'] - block['started_monotonic']] * len(rows))
                else:
                    times.extend(row['elapsed_seconds'] for row in rows)
            tokens.extend(intervals(block))
            if 'result' in block:
                tokens.extend(intervals(read(path.parent / block['attempt'] / 'generation-batches.json')))
            if 'whole_block_seconds' in block:
                tokens.extend(intervals(read(path.parent / block['attempt'] / 'execution/batches.json')))
        assert len(set(identities)) == len(identities)
        record = dict(run=directory.name, source=str(source.relative_to(ROOT)),
                      complete=source.exists(), tasks=len(identities), seconds=seconds,
                      amortized_seconds=seconds / len(identities), model_calls=model_calls,
                      mean_sample_seconds=statistics.mean(times) if times else None,
                      timed_samples=len(times), median_itl_ms=statistics.median(tokens) * 1000,
                      p95_itl_ms=statistics.quantiles(tokens, n=100)[94] * 1000,
                      max_itl_ms=max(tokens) * 1000,
                      itl_intervals_over_100ms=sum(t >= .1 for t in tokens))
        observations.append(record)
        print(json.dumps(record), flush=True)
    summaries = [ROOT / 'runs/official-lats-mbpp500-001/summary.json']
    summaries += sorted(ROOT.glob('runs/baselines/latentmas/runs/formal-*/summary.json'))
    for source in summaries:
        data = read(source)
        observations.append(dict(run=str(source.parent.relative_to(ROOT)), source=str(source.relative_to(ROOT)),
                                 complete=True, tasks=data['tasks'], seconds=data['seconds'],
                                 amortized_seconds=data['seconds'] / data['tasks'],
                                 mean_sample_seconds=data['mean_request_to_answer_seconds'],
                                 timed_samples=data['tasks'], median_itl_ms=data['itl_median_seconds'] * 1000,
                                 p95_itl_ms=data['itl_p95_seconds'] * 1000,
                                 max_itl_ms=data['itl_max_seconds'] * 1000,
                                 itl_intervals_over_100ms=data['itl_over_100ms']))
    source = ROOT / 'runs/source-interfaces-evaluation64-001/evaluation.json'
    data = read(source)
    for method in ['streamed', 'direct']:
        arm = data['methods'][method]
        observations.append(dict(run='source-interfaces-evaluation64-001/' + method,
                                 source=str(source.relative_to(ROOT)), complete=True, tasks=len(arm['queries']),
                                 seconds=arm['elapsed_seconds'],
                                 amortized_seconds=arm['elapsed_seconds'] / len(arm['queries']),
                                 mean_sample_seconds=statistics.mean(q['elapsed_seconds'] for q in arm['queries']),
                                 timed_samples=len(arm['queries']),
                                 max_itl_ms=max(q['maximum_itl_seconds'] for q in arm['queries']) * 1000,
                                 itl_intervals_over_100ms=arm['intervals_over_100ms']))
    output = ROOT / 'results/runtime_inventory'
    output.with_suffix('.json').write_text(json.dumps({
        'model': 'Qwen3.5-4B',
        'scope': 'Historical configurations, not a matched-budget comparison. Complete blocks only; excludes startup, warmup, interrupted work and offline scoring. Per-sample latency includes queueing within its active block. Amortized time is not latency. Missing measurements remain null.',
        'datasets': read(ROOT / 'configs/baselines/common/matrix.json')['datasets'],
        'observations': observations}, indent=2) + '\n')
    columns = list(dict.fromkeys(key for row in observations for key in row))
    with output.with_suffix('.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(observations)


if __name__ == '__main__':
    main()
