"""Execute a declared experiment list sequentially on each assigned GPU."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess

from project_paths import ROOT, log_path

def run_gpu(jobs, helper):
    codes = {}
    for job in jobs:
        output = Path(job['output'])
        output.mkdir(parents=True, exist_ok=True)
        with log_path(output, 'process.log').open('a') as log:
            result = subprocess.run(['bash', str(helper), str(job['gpu']),
                                     *job['command']], stdout=log, stderr=subprocess.STDOUT)
        codes[job['name']] = result.returncode
        print(json.dumps({'name': job['name'], 'exit_code': result.returncode}), flush=True)
        if result.returncode:
            break
    return codes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    groups = [[job for job in plan['jobs'] if job['gpu'] == gpu]
              for gpu in sorted({job['gpu'] for job in plan['jobs']})]
    with ThreadPoolExecutor(max_workers=len(groups)) as pool:
        results = list(pool.map(lambda jobs: run_gpu(jobs, plan['helper']), groups))
    report = ROOT / 'results/schedules' / (args.plan.stem + '.exit-codes.json')
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(results, indent=2) + '\n')


if __name__ == '__main__':
    main()
