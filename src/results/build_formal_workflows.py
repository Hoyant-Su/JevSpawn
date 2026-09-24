from collections import Counter
import json
from pathlib import Path


ROOT = Path(__file__).parents[2]
OUTPUT = ROOT / 'manuscript/tables'


def main():
    records = []
    for method, directory, citation in [
        ('FoldAgent', 'official-foldagent-bright104-002', 'sun2026scaling'),
        ('HiAgent', 'official-hiagent-bright104-001', 'hu-etal-2025-hiagent'),
    ]:
        run = ROOT / 'runs' / directory
        evaluation = json.loads((run / 'evaluation.json').read_text())
        protocol = json.loads((run / 'protocol.json').read_text())
        outcomes = [row for path in sorted(run.glob('block-*/complete.json'))
                    for row in json.loads(path.read_text())['results']]
        assert [row['task_id'] for row in outcomes] == [row['task_id'] for row in protocol['collections']]
        assert len(outcomes) == evaluation['queries'] == 104
        records.append({'method': method, 'citation': citation,
                        'source': str((run / 'evaluation.json').relative_to(ROOT)),
                        'status_counts': dict(Counter(row['status'] for row in outcomes)),
                        'errors': dict(Counter(row['error'] for row in outcomes if row['status'] == 'failed')),
                        **{key: evaluation[key] for key in ['queries', 'valid', 'ndcg', 'recall',
                            'whole_block_seconds', 'model_calls', 'output_tokens', 'truncated_calls',
                            'peak_allocated_bytes', 'max_itl_ms', 'intervals_over_100ms']}})
    lines = [r'\begin{table}[t]', r'\centering', r'\small', r'\resizebox{\linewidth}{!}{%',
             r'\begin{tabular}{lrrrrrrr}', r'\toprule',
             r'Method & Valid / 104 & nDCG@10 & Recall@10 & Workload (s) & Calls & Peak GiB & Max ITL (ms) \\',
             r'\midrule']
    for row in records:
        values = [row['method'] + r' \cite{' + row['citation'] + '}', str(row['valid']),
                  f"{row['ndcg']:.4f}", f"{row['recall']:.4f}", f"{row['whole_block_seconds']:.2f}",
                  str(row['model_calls']), f"{row['peak_allocated_bytes'] / 2**30:.2f}",
                  f"{row['max_itl_ms']:.2f}"]
        lines.append(' & '.join(values) + r' \\')
    lines.extend([r'\bottomrule', r'\end{tabular}}',
        r'\caption{Document collection results on 104 BRIGHT Pony queries with 128 candidates per query and a shared Qwen3.5-4B service. Both methods use an input capacity of 8,192 tokens and at most 512 generated tokens per call. All assigned queries contribute to quality and workload time. Valid rankings contain ten distinct candidate identifiers. Failed and invalid rankings receive zero task score. Method budgets and failure breakdowns appear in Appendix~\ref{sec:setup}.}',
        r'\label{tab:published-bright}', r'\end{table}'])
    (OUTPUT / 'published_bright.tex').write_text('\n'.join(lines) + '\n')
    (OUTPUT / 'published_bright.json').write_text(json.dumps({'records': records}, indent=2) + '\n')
    path = ROOT / 'runs/official-lats-mbpp500-001/summary.json'
    summary = json.loads(path.read_text())
    assert summary['tasks'] == 500 and summary['blocks'] == 63
    (OUTPUT / 'published_lats.json').write_text(json.dumps({
        'source': str(path.relative_to(ROOT)), 'citation': 'zhou2024lats', **summary}, indent=2) + '\n')


if __name__ == '__main__':
    main()
