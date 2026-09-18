"""Launch one batch worker per explicitly assigned GPU on the selected instance."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--entrypoint", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    project = Path(__file__).resolve().parents[1]
    root = project.parents[1]
    directory = Path(config["run_dir"])
    directory.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ, PYTHONPATH=str(project / "src"), PYTHONDONTWRITEBYTECODE="1",
                       HF_HOME=str(root / "runtime/cache/huggingface"),
                       TRITON_CACHE_DIR=str(root / "runtime/cache/triton"),
                       XDG_CACHE_HOME=str(root / "runtime/cache"),
                       TORCHINDUCTOR_CACHE_DIR=str(root / "runtime/cache/torchinductor"),
                       TMPDIR=str(root / "runtime/tmp"), TOKENIZERS_PARALLELISM="false",
                       HF_HUB_OFFLINE="1", HF_HUB_DISABLE_IMPLICIT_TOKEN="1")
    for key in ("HF_HOME", "TRITON_CACHE_DIR", "XDG_CACHE_HOME", "TORCHINDUCTOR_CACHE_DIR", "TMPDIR"):
        Path(environment[key]).mkdir(parents=True, exist_ok=True)
    processes = []
    for rank in range(config["world_size"]):
        with (directory / f"worker-{rank}.log").open("a") as log:
            entrypoint = [str(args.entrypoint.resolve())] if args.entrypoint else ["-m", "jev_spawn.run"]
            command = [str(args.python), "-u", *entrypoint, "--config", str(args.config.resolve()), "--rank", str(rank)]
            process = subprocess.Popen(command, env=dict(environment, CUDA_VISIBLE_DEVICES=str(rank)), stdout=log, stderr=subprocess.STDOUT)
            processes.append(process)
    (directory / "launcher.json").write_text(json.dumps({"python": str(args.python), "pids": [p.pid for p in processes]}, indent=2) + "\n")
    codes = [process.wait() for process in processes]
    print(json.dumps({"worker_exit_codes": codes}), flush=True)
    sys.exit(int(any(codes)))


if __name__ == "__main__":
    main()
