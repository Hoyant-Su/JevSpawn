import csv
import json
from pathlib import Path


ROOT = Path(__file__).parents[2]


def main():
    records = []
    for name, directory, citation in [
        ('ReAct', 'official-react-aqua-test-001', 'yao2023react'),
        ('LLMCompiler', 'official-compiler-aqua-test-001', 'kim2024llmcompiler'),
    ]:
        path = ROOT / 'runs' / directory / 'evaluation.json'
        result = json.loads(path.read_text())
        records.append({'method': name, 'citation': citation, 'source': str(path.relative_to(ROOT)),
                        **{key: result[key] for key in ['tasks', 'correct', 'valid', 'accuracy',
                            'elapsed_seconds', 'output_tokens', 'peak_allocated_bytes',
                            'max_itl_ms', 'intervals_over_100ms', 'budget']}})
    path = ROOT / 'runs/baselines/latentmas/runs/formal-aqua254-001/summary.json'
    result = json.loads(path.read_text())
    records.append({'method': 'LatentMAS', 'citation': 'zou2026latent',
                    'source': str(path.relative_to(ROOT)), 'tasks': result['tasks'],
                    'correct': result['correct'], 'valid': result['valid_answers'],
                    'accuracy': result['accuracy'], 'elapsed_seconds': result['seconds'],
                    'output_tokens': result['generated_tokens'],
                    'peak_allocated_bytes': result['peak_allocated_bytes'],
                    'max_itl_ms': result['itl_max_seconds'] * 1000,
                    'intervals_over_100ms': result['itl_over_100ms'],
                    'budget': {'latent_roles': 3, 'steps_per_role': result['latent_steps_per_role'],
                               'final_token_budget': result['final_token_budget']}})
    path = ROOT / 'runs/official-agentprune-aqua254-001/evaluation.json'
    result = json.loads(path.read_text())
    settings = json.loads(path.with_name('settings.json').read_text())
    records.append({'method': 'AgentPrune', 'citation': 'zhang2025agentprune',
                    'source': str(path.relative_to(ROOT)),
                    **{key: result[key] for key in ['tasks', 'correct', 'valid', 'accuracy',
                        'elapsed_seconds', 'output_tokens', 'peak_allocated_bytes',
                        'max_itl_ms', 'intervals_over_100ms', 'training_provenance']},
                    'budget': {key: settings[key] for key in ['num_rounds', 'generation']}})
    path = ROOT / 'runs/official-dyflow-aqua254-001/evaluation.json'
    result = json.loads(path.read_text())
    records.append({'method': 'DyFlow', 'citation': 'wang2025dyflow',
                    'source': str(path.relative_to(ROOT)),
                    **{key: result[key] for key in ['tasks', 'correct', 'valid', 'accuracy',
                        'elapsed_seconds', 'output_tokens', 'peak_allocated_bytes',
                        'max_itl_ms', 'intervals_over_100ms', 'budget',
                        'adapter_exceptions', 'model_request_errors']}})
    assert all(row['tasks'] == 254 for row in records)
    output = ROOT / 'manuscript/tables'
    headers = ['Method', 'Correct / 254', 'Accuracy (%)', 'Valid / 254', 'Workload (s)',
               'Generated tokens', 'Peak GiB', 'Max ITL (ms)']
    rows = [[row['method'], row['correct'], f"{100 * row['accuracy']:.2f}", row['valid'],
             f"{row['elapsed_seconds']:.2f}", row['output_tokens'],
             f"{row['peak_allocated_bytes'] / 2**30:.2f}", f"{row['max_itl_ms']:.2f}"]
            for row in records]
    with (output / 'published_aqua.csv').open('w') as stream:
        writer = csv.writer(stream)
        writer.writerow(headers)
        writer.writerows(rows)
    lines = [r'\begin{table}[t]', r'\centering', r'\small', r'\resizebox{\linewidth}{!}{%',
             r'\begin{tabular}{lrrrrrrr}', r'\toprule',
             ' & '.join(value.replace('%', r'\%') for value in headers) + r' \\', r'\midrule']
    for record, row in zip(records, rows):
        row[0] += r' \cite{' + record['citation'] + '}'
        lines.append(' & '.join(map(str, row)) + r' \\')
    lines.extend([r'\bottomrule', r'\end{tabular}}',
        r'\caption{Published agent methods on 254 AQuA questions with Qwen3.5-4B and a common request batch cap. Workload time includes complete reasoning and tool execution under the method budgets specified in Appendix~\ref{sec:setup}. Startup and topology learning are reported separately. Invalid answers remain incorrect. Peak memory includes model weights.}',
        r'\label{tab:published-aqua}', r'\end{table}'])
    (output / 'published_aqua.tex').write_text('\n'.join(lines) + '\n')
    (output / 'published_aqua.json').write_text(json.dumps({'records': records}, indent=2) + '\n')


if __name__ == '__main__':
    main()
