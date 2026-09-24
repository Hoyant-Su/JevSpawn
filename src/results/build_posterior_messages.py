import json
from pathlib import Path


root = Path(__file__).resolve().parents[2]
sources = [('Source', 'runs/posterior-messages-development-001/evaluation.json'),
           ('Anonymous', 'runs/posterior-messages-anonymous-001/evaluation.json')]
lines = [r'\begin{table}[t]', r'\centering\small', r'\begin{tabular}{llrrr}', r'\toprule',
         r'Identity & Message & nDCG@10 & Time & Memory \\', r'\midrule']
rows, comparisons = [], []
for identity, path in sources:
    report = json.loads((root / path).read_text())
    comparisons.append({'identity': identity, 'source': path, 'paired': report['paired']})
    for arm, name in [('winner_only', 'Selected options'), ('full_posterior', 'Options and probabilities')]:
        value = report['methods'][arm]
        lines.append(f'{identity} & {name} & {value["ndcg"]:.4f} & {value["elapsed_seconds"]:.2f} & '
                     f'{value["peak_allocated_bytes"] / 2**30:.2f}' + r' \\')
        rows.append({'identity': identity, 'arm': arm,
                     **{key: item for key, item in value.items() if key != 'queries'}})
lines.extend([r'\bottomrule', r'\end{tabular}',
    r'\caption{Structured communication on eight BRIGHT development queries with 128 documents each. Generated questions and evidence worker outputs are fixed. A finite receiver reads selected options or complete candidate probabilities without document text. Source identities contain topic words, whereas anonymous identities use integer indices. Time is receiver execution in seconds and memory is peak allocated GiB including weights. Both identity conditions receive full warmup and use Qwen3.5-4B, BF16, one H100, and a batch capacity of eight. Upstream planning and evidence computation are excluded.}',
    r'\label{tab:posterior-messages}', r'\end{table}'])
target = root / 'manuscript/tables/posterior_messages'
target.with_suffix('.tex').write_text('\n'.join(lines) + '\n')
target.with_suffix('.json').write_text(json.dumps({'rows': rows, 'comparisons': comparisons}, indent=2) + '\n')
print(json.dumps({'rows': rows, 'comparisons': comparisons}))
