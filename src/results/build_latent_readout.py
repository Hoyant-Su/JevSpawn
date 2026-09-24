import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads((args.run / 'summary.json').read_text())
    protocol = json.loads((args.run / 'protocol.json').read_text())
    expected = [task['task_id'] for task in protocol['tasks']]
    assert len(expected) == len(set(expected)) == 32
    records = [json.loads(p.read_text()) for p in sorted((args.run / 'measured').glob('block-*.json'))]
    names = {'latent-0': 'Direct readout', 'latent-10': 'Latent recurrence, 10 steps',
             'latent-30': 'Latent recurrence, 30 steps', 'text': 'Generated reasoning'}
    rows = []
    lines = [r'\begin{table}[t]', r'\centering\small', r'\begin{tabular}{lrrrrr}',
             r'\toprule', r'Computation & AQuA & MATH & Valid & Time & Memory \\', r'\midrule']
    for name, title in names.items():
        chosen = [b for b in records if (f"latent-{b['condition']['steps']}" if b['condition']['mode'] == 'latent' else 'text') == name]
        assert [r['task_id'] for b in chosen for r in b['predictions']] == expected
        assert all(b['batch_size'] == 8 for b in chosen)
        value = summary['conditions'][name]
        assert value['compute_seconds'] == sum(b['compute_seconds'] for b in chosen)
        correct = {dataset: result['correct'] for dataset, result in value['datasets'].items()}
        valid = sum(result['valid'] for result in value['datasets'].values())
        zero = [r['correct']['latent-0'] for r in summary['paired_correctness']]
        current = [r['correct'][name] for r in summary['paired_correctness']]
        rows.append({'condition': name, **value,
                     'corrected_vs_direct': sum(a and not b for a, b in zip(current, zero)),
                     'regressed_vs_direct': sum(b and not a for a, b in zip(current, zero))})
        lines.append(f"{title} & {correct['aqua']}/16 & {correct['processbench_gsm8k']}/16 & "
                     f"{valid}/32 & {value['compute_seconds']:.2f} & {value['peak_allocated_bytes'] / 2**30:.2f} " + r'\\')
    lines += [r'\bottomrule', r'\end{tabular}',
              r'\caption{Computation before finite candidate readout on 16 AQuA development questions and '
              r'16 ProcessBench MATH development solutions. All conditions use Qwen3.5-4B, BF16, '
              r'batches of eight, identical task prompts, and the same final readout suffix. '
              r'Latent computation uses the released LatentMAS recurrence~\cite{zou2026latent}. '
              r'Generated reasoning permits 2,048 tokens and requires normal termination. '
              r'Accuracy columns give correct counts. Time includes complete measured inference in seconds '
              r'after a full warmup, and memory is peak allocated GiB.}',
              r'\label{tab:latent-readout}', r'\end{table}']
    args.output.with_suffix('.tex').write_text('\n'.join(lines) + '\n')
    args.output.with_suffix('.json').write_text(json.dumps({'source': str(args.run / 'summary.json'),
        'scope': summary['scope'], 'records': rows, 'promotion': summary['promotion']}, indent=2) + '\n')
    print(json.dumps({'outcomes': summary['outputs'], 'promoted': any(v['passes'] for v in summary['promotion'].values())}))


if __name__ == '__main__':
    main()
