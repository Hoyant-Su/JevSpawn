import json
from pathlib import Path
import statistics


root = Path(__file__).resolve().parents[2]
source = root / 'runs/evidence-interfaces-development-001/evaluation.json'
report = json.loads(source.read_text())
methods = [('streamed', 'Generated interfaces, streamed'),
           ('tiled_independent', 'Generated interfaces, independent'),
           ('json', 'Generated interfaces, JSON'), ('direct', 'Direct relevance')]
lines = [r'\begin{table}[t]', r'\centering\small', r'\begin{tabular}{lrrrrr}', r'\toprule',
         r'Method & nDCG & Valid & Time & Memory & Agents \\', r'\midrule']
rows = []
for key, name in methods:
    value = report['methods'][key]
    tasks = value['queries']
    assert len(tasks) == 8
    agents = statistics.mean(row['agent_invocations'] for row in tasks)
    lines.append(f'{name} & {value["ndcg"]:.4f} & {value["valid"]}/8 & '
                 f'{value["elapsed_seconds"]:.2f} & {value["peak_allocated_bytes"] / 2**30:.2f} & {agents:.2f}' + r' \\')
    rows.append({'method': key, **{k: v for k, v in value.items() if k != 'queries'},
                 'mean_agent_invocations': agents,
                 'maximum_dependency_depth': max(row['model_dependency_depth'] for row in tasks)})
lines.extend([r'\bottomrule', r'\end{tabular}',
              r'\caption{Generated evidence interfaces on eight BRIGHT development queries, each with 128 original candidate documents. nDCG is evaluated at ten. Time is the complete measured workload in seconds after a full warmup, including planning, workers, communication and ranking. Memory is maximum allocated GiB including weights. Agents denotes the mean number of model invocations per query, including planning and receiving agents. Each method uses one H100 and a worker batch capacity of eight. The separate terminal message interventions are excluded from workload time.}',
              r'\label{tab:evidence-interfaces}', r'\end{table}'])
target = root / 'manuscript/tables/evidence_interfaces'
target.with_suffix('.tex').write_text('\n'.join(lines) + '\n')
target.with_suffix('.json').write_text(json.dumps({'source': str(source.relative_to(root)), 'rows': rows,
      'message_utility': report['mean_message_utility'], 'useful_refinement_cases': report['useful_refinement_cases']}, indent=2) + '\n')
print(json.dumps(rows))
