import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--tests', type=Path, required=True)
    parser.add_argument('--timeout-seconds', type=float, required=True)
    args = parser.parse_args()
    protocol = json.loads((args.run / 'protocol.json').read_text())
    settings = protocol['settings']
    report = {'task_ids': protocol['task_ids'], 'offline_timeout_seconds': args.timeout_seconds, 'stages': {}}
    for stage in ['warmup', 'measured']:
        directory = args.run / stage
        summary = json.loads((directory / 'summary.json').read_text())
        batches = json.loads((directory / 'batches.json').read_text())
        rows = [json.loads(path.read_text()) for path in sorted(directory.glob('MBPP_*/result.json'))]
        assert {row['task_id'] for row in rows} == set(protocol['task_ids'])
        subprocess.run([sys.executable, settings['evaluation_script'], '--solutions',
                        str(directory / 'solutions.jsonl'), '--tests', str(args.tests), '--sandbox',
                        settings['sandbox'], '--output', str(directory / 'heldout.jsonl'), '--work-dir',
                        str(directory / 'heldout-tools'), '--workers', str(settings['task_count']),
                        '--timeout', str(args.timeout_seconds), '--memory-mb', str(settings['sandbox_memory_mb'])], check=True)
        scores = [json.loads(line) for line in (directory / 'heldout.jsonl').read_text().splitlines()]
        summary.update(status_counts=dict(Counter(row['status'] for row in rows)),
                       public_passes=sum(row.get('public_test_passed', False) for row in rows),
                       heldout_passes=sum(row['status'] == 'passed' for row in scores),
                       quality_denominator=len(rows),
                       actual_model_sequences=sum(batch['batch_size'] for batch in batches),
                       input_tokens=sum(sum(batch['input_tokens']) for batch in batches),
                       output_tokens=sum(sum(batch['output_tokens']) for batch in batches),
                       truncated_model_sequences=sum(sum(batch['truncated']) for batch in batches),
                       peak_allocated_bytes=max(batch['peak_allocated_bytes'] for batch in batches),
                       peak_reserved_bytes=max(batch['peak_reserved_bytes'] for batch in batches),
                       public_evaluation_calls=sum(len(row['public_evaluations']) for row in rows),
                       node_event_counts=dict(Counter(event['function'] for row in rows for event in row['node_events'])),
                       max_observed_depth=max((event['depth'] for row in rows for event in row['node_events']), default=0),
                       failures=[{'task_id': row['task_id'], 'error': row['error']} for row in rows if row['status'] == 'failed'])
        report['stages'][stage] = summary
    (args.run / 'qualification-summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
