import argparse
import collections
import concurrent.futures
import json
import math
from pathlib import Path
import subprocess
import tempfile
import time
import uuid


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sandbox_command(sandbox, program):
    command = [str(sandbox), "--unshare-all", "--die-with-parent", "--new-session", "--clearenv", "--cap-drop", "ALL"]
    for path in ["/usr", "/lib", "/lib64"]:
        command.extend(["--ro-bind", path, path])
    return command + [
        "--dir", "/dev", "--ro-bind", "/dev/null", "/dev/null",
        "--ro-bind", "/dev/urandom", "/dev/urandom", "--dir", "/tmp", "--dir", "/work",
        "--ro-bind", str(program), "/work/check.py", "--chdir", "/work",
        "--remount-ro", "/",
        "--setenv", "HOME", "/tmp", "--setenv", "PATH", "/usr/bin",
        "--setenv", "PYTHONHASHSEED", "0", "/usr/bin/python3", "-I", "-B", "/work/check.py",
    ]


def evaluate_one(sample, test, args):
    started = time.monotonic()
    marker = "EVALUATION_PASSED_" + uuid.uuid4().hex
    limits = [("RLIMIT_AS", args.memory_mb * 1024 * 1024), ("RLIMIT_CPU", math.ceil(args.timeout)), ("RLIMIT_FSIZE", 1024 * 1024), ("RLIMIT_NPROC", 32), ("RLIMIT_NOFILE", 64), ("RLIMIT_CORE", 0)]
    setup = "import resource\n" + "\n".join(f"resource.setrlimit(resource.{name}, ({value}, {value}))" for name, value in limits)
    program_text = (
        setup + "\nnamespace = {'__name__': '__main__'}\n"
        + f"exec(compile({sample['solution']!r}, 'solution.py', 'exec'), namespace)\n"
        + f"exec(compile({test['test_code']!r}, 'tests.py', 'exec'), namespace)\n"
        + f"print({marker!r}, flush=True)\n"
    )
    path = Path(tempfile.mkdtemp(prefix="evaluation-", dir=args.work_dir))
    program = path / "check.py"
    program.write_text(program_text)
    with (path / "stdout").open("wb") as stdout, (path / "stderr").open("wb") as stderr:
        try:
            process = subprocess.run(sandbox_command(args.sandbox, program), stdout=stdout, stderr=stderr, timeout=args.timeout + 2)
            status = "passed" if process.returncode == 0 and (path / "stdout").read_text(errors="replace").rstrip().endswith(marker) else "failed"
            returncode = process.returncode
        except subprocess.TimeoutExpired:
            status = "timeout"
            returncode = None
    diagnostic = (path / "stderr").read_text(errors="replace")[-4000:]
    return {"task_id": sample["task_id"], "dataset": test["dataset"], "status": status, "returncode": returncode, "elapsed_seconds": time.monotonic() - started, "diagnostic": diagnostic, "artifact_directory": str(path)}


def main():
    parser = argparse.ArgumentParser(description="Evaluate frozen complete Python solutions in an explicit bubblewrap sandbox. Original tests only, not EvalPlus.")
    parser.add_argument("--solutions", type=Path, required=True)
    parser.add_argument("--tests", type=Path, required=True)
    parser.add_argument("--sandbox", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=10)
    parser.add_argument("--memory-mb", type=int, default=512)
    args = parser.parse_args()
    args.sandbox = args.sandbox.resolve(strict=True)
    args.work_dir = args.work_dir.resolve()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    samples = read_jsonl(args.solutions)
    if len({sample["task_id"] for sample in samples}) != len(samples):
        parser.error("Solutions must contain exactly one final record per task_id.")
    tests = {row["task_id"]: row for row in read_jsonl(args.tests)}
    unknown = {sample["task_id"] for sample in samples} - tests.keys()
    if unknown:
        parser.error(f"No evaluation test for task IDs: {sorted(unknown)}")
    program = Path(tempfile.mkdtemp(prefix="preflight-", dir=args.work_dir)) / "check.py"
    program.write_text("assert __import__('os').getcwd() == '/work'\n")
    subprocess.run(sandbox_command(args.sandbox, program), check=True, timeout=10)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(lambda sample: evaluate_one(sample, tests[sample["task_id"]], args), samples))
    with args.output.open("w") as stream:
        for result in results:
            stream.write(json.dumps(result) + "\n")
    by_dataset = collections.defaultdict(collections.Counter)
    for result in results:
        by_dataset[result["dataset"]][result["status"]] += 1
    summary = {
        "sandbox": "bubblewrap: user/pid/network/mount namespaces, read-only root and runtime mounts, no workspace/home mounts or writable directories, cleared environment, resource limits; artifacts retained without cleanup",
        "evaluation": "Original HumanEval/MBPP assertions only. Not HumanEval+ or MBPP+. One frozen final solution per task.",
        "counts": dict(collections.Counter(result["status"] for result in results)),
        "datasets": {dataset: {"total": sum(counts.values()), "passed": counts["passed"], "final_solution_pass_rate": counts["passed"] / sum(counts.values())} for dataset, counts in by_dataset.items()},
    }
    args.output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
