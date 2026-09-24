import json
from pathlib import Path


ROOT = Path(__file__).parents[2]


def main():
    records = []
    for directory, methods in [
        ('structured-flow-constrained-development-001',
         [('graph', 'Constrained graph, 4B'), ('direct', 'Constrained direct, 4B')]),
        ('structured-flow-constrained-development-27b-001',
         [('graph', 'Constrained graph, 27B'), ('direct', 'Constrained direct, 27B')]),
        ('structured-flow-typed-development-27b-001',
         [('graph', 'Typed graph, 27B'), ('direct', 'Typed direct, 27B')]),
        ('structured-flow-lexical-development-27b-001',
         [('graph', 'Local reference graph, 27B'), ('direct', 'Local reference direct, 27B'),
          ('no_messages', 'Finalizer only, 27B')]),
        ('structured-flow-grounded-development-27b-001',
         [('graph', 'Source bound graph, 27B'), ('direct', 'Source bound direct, 27B'),
          ('no_messages', 'Source bound finalizer only, 27B')]),
    ]:
        source = ROOT / 'runs' / directory / 'qualification_summary.json'
        summary = json.loads(source.read_text())
        for method, name in methods:
            result = summary['methods'][method]
            assert result['assigned_tasks'] == 8
            records.append({'configuration': name, 'source': str(source.relative_to(ROOT)),
                            **{key: result[key] for key in ['assigned_tasks', 'valid', 'correct',
                                'mean_completed_model_requests_per_assigned_task',
                                'maximum_completed_model_requests_per_task',
                                'complete_workload_compute_seconds']},
                            'planner_seconds': sum(row.get('model_seconds_by_operation', {}).get('expand', 0)
                                                   for row in result['tasks']),
                            'max_itl_ms': max(row['max_itl_ms'] for row in result['tasks'])})
    lines = [r'\begin{table}[t]', r'\centering', r'\small', r'\resizebox{\linewidth}{!}{%',
             r'\begin{tabular}{lrrrrrr}', r'\toprule',
             r'Configuration & Valid / 8 & Correct / 8 & Mean calls & Max calls & Workload (s) & Planning (s) \\',
             r'\midrule']
    for row in records:
        values = [row['configuration'], str(row['valid']), str(row['correct']),
                  f"{row['mean_completed_model_requests_per_assigned_task']:.3f}",
                  str(row['maximum_completed_model_requests_per_task']),
                  f"{row['complete_workload_compute_seconds']:.2f}", f"{row['planner_seconds']:.2f}"]
        lines.append(' & '.join(values) + r' \\')
    lines.extend([r'\bottomrule', r'\end{tabular}}',
        r'\caption{Generated computation graphs on eight development tasks, with two tasks each from coding, mathematical error localization, scientific claim verification and logic grids. Each graph configuration is compared with direct answering by the same model under the original answer schema. Qwen3.5-4B and Qwen3.8-27B differ in both size and model generation. Tasks execute individually with at most eight active model requests and 4,096 tokens per generation. Calls include returned model requests with invalid outputs and exclude deterministic operations. Workload time includes failed tasks. Typed graphs introduce model defined intermediate schemas and a final answer agent. Local references define dependencies through earlier values. The finalizer only control uses the same final answer agent without planning or intermediate messages.}',
        r'\label{tab:graph-qualification}', r'\end{table}'])
    output = ROOT / 'manuscript/tables'
    (output / 'graph_qualification.tex').write_text('\n'.join(lines) + '\n')
    (output / 'graph_qualification.json').write_text(json.dumps({'records': records}, indent=2) + '\n')


if __name__ == '__main__':
    main()
