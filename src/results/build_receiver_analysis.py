import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', type=Path, required=True)
    args = parser.parse_args()
    stage = json.loads(args.stage.read_text())
    source = Path(stage['output']) / 'evaluation.json'
    report = json.loads(source.read_text())
    lines = [r'\begin{table}[t]', r'\centering\small', r'\begin{tabular}{llrrr}',
             r'\toprule', r'Evidence policy & Receiver output & EM & F1 & Time \\', r'\midrule']
    for arm, name in [('streamed', 'Streamed selection'), ('json', 'JSON selection'), ('direct', 'Direct RAG')]:
        result = report['methods'][arm]
        for mode, label, timing in [('original', 'Rationale and answer', 'original_receiver_call_seconds'),
                                    ('answer_only', 'Answer only', 'receiver_call_seconds')]:
            metrics = result['metrics'][mode]
            lines.append(f'{name} & {label} & {100 * metrics["exact_match"]:.1f} & '
                         f'{100 * metrics["token_f1"]:.1f} & {result[timing]:.2f}' + r' \\')
    caption = (
        f'Final answer generation on {report["task_count"]} MultiHop-RAG development questions. '
        'Within each evidence policy, both receivers observe the same passages in the same order. '
        'The answer only condition removes the required rationale from the output schema and instruction. '
        f'Both conditions retain an output budget of {stage["fixed"]["final_tokens"]} tokens and return evidence indices. '
        'EM and token F1 are percentages. Time is the sum of receiver call durations in seconds, '
        'excluding planning and evidence selection. Each condition has one measured pass after full workload warmup.')
    lines.extend([r'\bottomrule', r'\end{tabular}', r'\caption{' + caption + '}',
                  r'\label{tab:receiver-reasoning}', r'\end{table}'])
    target = Path('manuscript/tables/receiver_reasoning')
    target.with_suffix('.tex').write_text('\n'.join(lines) + '\n')
    target.with_suffix('.json').write_text(json.dumps({'source': str(source), 'report': report}, indent=2) + '\n')


if __name__ == '__main__':
    main()
