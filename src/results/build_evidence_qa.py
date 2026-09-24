import json
from pathlib import Path


root = Path(__file__).resolve().parents[2]
source = root / 'runs/evidence-qa-development-001/evaluation.json'
report = json.loads(source.read_text())
lines = [r'\begin{table}[t]', r'\centering\small',
         r'\begin{tabular}{lrrrrrr}', r'\toprule',
         r'Method & Valid & EM & F1 & Time & Memory & Workers \\', r'\midrule']
rows = []
for arm, name in [('direct', 'Direct RAG'), ('tiled_independent', 'Independent finite'),
                  ('streamed', 'Streamed finite'), ('json', 'JSON workers')]:
    value = report['methods'][arm]
    metrics = value['metrics']
    lines.append(f'{name} & {value["completed"]}/8 & {100 * metrics["exact_match"]:.1f} & '
                 f'{100 * metrics["token_f1"]:.1f} & {value["elapsed_seconds"]:.2f} & '
                 f'{value["peak_allocated_bytes"] / 2**30:.2f} & {value["mean_workers"]:.1f}' + r' \\')
    rows.append({'arm': arm, **{key: item for key, item in value.items() if key != 'queries'}})
lines.extend([r'\bottomrule', r'\end{tabular}',
    r'\caption{Evidence conditioned interface generation on eight original MultiHop-RAG development questions. EM and token F1 are percentages over all questions, including invalid answers. Time includes retrieval, planning, workers, communication and answer generation, in seconds. Memory is peak allocated GiB including model weights. Workers is the mean number of passage and field judgments per question. All methods use Qwen3.5-4B, BF16 and one H100, with worker batch capacity eight and one query in flight. Direct RAG answers from the initial 16 retrieved passages. Structured policies can acquire additional passages and therefore differ in realized evidence. Each method receives a complete warmup and one measured pass.}',
    r'\label{tab:evidence-qa}', r'\end{table}'])
target = root / 'manuscript/tables/evidence_qa'
target.with_suffix('.tex').write_text('\n'.join(lines) + '\n')
target.with_suffix('.json').write_text(json.dumps({'source': str(source.relative_to(root)), 'rows': rows,
    'message_utility': report['mean_message_utility'], 'useful_refinement_cases': report['useful_refinement_cases'],
    'paired_exact_match_difference': report['paired_exact_match_difference'],
    'paired_exact_match_interval95': report['paired_exact_match_interval95']}, indent=2) + '\n')
print(json.dumps(rows))
