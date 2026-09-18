import argparse
import ast
import concurrent.futures
import copy
import datetime
import gzip
import json
from pathlib import Path
import urllib.request


HUMANEVAL_COMMIT = "6d43fb980f9fee3c892a914eda09951f772ad10d"
MBPP_COMMIT = "f82046ba5aabbbb427dbfd38a254d26bff08b533"
MBPP_CARD_COMMIT = "4bb6404fdc6cacfda99d4ac4205087b89d32030c"
HUMANEVAL_ROOT = f"https://raw.githubusercontent.com/openai/human-eval/{HUMANEVAL_COMMIT}"
MBPP_ROOT = f"https://raw.githubusercontent.com/google-research/google-research/{MBPP_COMMIT}"
SOURCES = {
    "HumanEval.jsonl.gz": f"{HUMANEVAL_ROOT}/data/HumanEval.jsonl.gz",
    "HumanEval.LICENSE": f"{HUMANEVAL_ROOT}/LICENSE",
    "mbpp.jsonl": f"{MBPP_ROOT}/mbpp/mbpp.jsonl",
    "mbpp.source-readme.txt": f"{MBPP_ROOT}/mbpp/README.md",
    "google-research.LICENSE": f"{MBPP_ROOT}/LICENSE",
    "mbpp.dataset-card.txt": f"https://huggingface.co/datasets/google-research-datasets/mbpp/raw/{MBPP_CARD_COMMIT}/README.md",
}
MBPP_SPLITS = {"prompting": range(1, 11), "test": range(11, 511), "validation": range(511, 601), "train": range(601, 975)}


def download(item, destination):
    name, url = item
    request = urllib.request.Request(url, headers={"User-Agent": "jev-spawn-data"})
    with urllib.request.urlopen(request, timeout=30) as response:
        content = response.read()
    (destination / name).write_bytes(content)
    return {"file": name, "url": url, "bytes": len(content)}


def reuse_sources(source_dir, destination):
    sources = []
    for name, url in SOURCES.items():
        content = (source_dir / name).read_bytes()
        (destination / name).write_bytes(content)
        sources.append({"file": name, "url": url, "bytes": len(content)})
    return sources


def interface_node(node):
    result = copy.deepcopy(node)
    result.decorator_list = []
    if isinstance(result, ast.ClassDef):
        result.body = [interface_node(child) for child in result.body if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
    else:
        result.body = []
    result.body = result.body or [ast.Expr(value=ast.Constant(value=Ellipsis))]
    return result


def write_jsonl(path, rows):
    with path.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Prepare isolated runtime tasks and evaluation-only tests from pinned official sources.")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[3] / "data" / "agent_spawn")
    parser.add_argument("--mbpp-split", choices=["test", "all"], default="test")
    parser.add_argument("--mbpp-examples", type=int, choices=[0, 1], default=0)
    parser.add_argument("--source-dir", type=Path, help="Reuse downloaded source files from this directory.")
    args = parser.parse_args()
    raw = args.output / "sources"
    evaluation = args.output / "evaluation"
    raw.mkdir(parents=True, exist_ok=True)
    evaluation.mkdir(parents=True, exist_ok=True)
    if args.source_dir:
        sources = reuse_sources(args.source_dir, raw)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(SOURCES)) as pool:
            sources = list(pool.map(lambda item: download(item, raw), SOURCES.items()))
    humaneval = [json.loads(line) for line in gzip.decompress((raw / "HumanEval.jsonl.gz").read_bytes()).decode().splitlines()]
    mbpp = [json.loads(line) for line in (raw / "mbpp.jsonl").read_text().splitlines()]
    mbpp = sorted((row for row in mbpp if args.mbpp_split == "all" or row["task_id"] in MBPP_SPLITS["test"]), key=lambda row: row["task_id"])
    split_ids = {split: [f"MBPP/{row['task_id']}" for row in mbpp if row["task_id"] in ids] for split, ids in MBPP_SPLITS.items()}
    tasks = []
    tests = []
    for row in humaneval:
        tasks.append({"task_id": row["task_id"], "dataset": "humaneval", "prompt": row["prompt"], "entry_point": row["entry_point"]})
        tests.append({"task_id": row["task_id"], "dataset": "humaneval", "test_code": row["test"] + f"\ncheck({row['entry_point']})\n"})
    for row in mbpp:
        declarations = [node for node in ast.parse(row["code"]).body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
        interface = "\n\n".join(ast.unparse(interface_node(node)) for node in declarations)
        task_id = f"MBPP/{row['task_id']}"
        prompt = row["text"] + "\n\nImplement the following Python interface. Return a complete Python module.\n\n" + interface
        if args.mbpp_examples:
            prompt += "\n\nPublic example assertion:\n" + row["test_list"][0]
        tasks.append({"task_id": task_id, "dataset": "mbpp", "prompt": prompt, "entry_point": ""})
        test_code = "\n".join([row["test_setup_code"], *row["test_list"][args.mbpp_examples:], *row["challenge_test_list"]])
        tests.append({"task_id": task_id, "dataset": "mbpp", "test_code": test_code})
    write_jsonl(args.output / "tasks.jsonl", tasks)
    write_jsonl(evaluation / "tests.jsonl", tests)
    manifest = {
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "sources": sources,
        "counts": {"humaneval": len(humaneval), "mbpp": len(mbpp), **{f"mbpp_{split}": len(ids) for split, ids in split_ids.items()}, "total": len(tasks), "mbpp_with_challenge_tests": sum(bool(row["challenge_test_list"]) for row in mbpp)},
        "mbpp_split": args.mbpp_split,
        "mbpp_examples": args.mbpp_examples,
        "mbpp_split_task_ids": split_ids,
        "licenses": {
            "humaneval": {"dataset": "MIT", "source": SOURCES["HumanEval.LICENSE"]},
            "mbpp": {"dataset": "CC-BY-4.0", "source": SOURCES["mbpp.dataset-card.txt"], "repository_code": "Apache-2.0"},
        },
        "protocol": {
            "humaneval": "Original HumanEval tests; not HumanEval+.",
            "mbpp": "Selected split: " + args.mbpp_split + f". Official split IDs are recorded separately. Statement plus reference interface signatures and {args.mbpp_examples} public example assertion. The first {args.mbpp_examples} original test_list assertions are exposed in source order and excluded from evaluation; remaining test_list and all challenge_test_list assertions are held out. No result-dependent example selection. This is an explicit prompt variant, not a standard MBPP+ score.",
            "workload": "The all-splits selection is a bulk coding workload, not a standardized MBPP benchmark. Report official MBPP test IDs 11-510 separately.",
            "generation_input": f"Only tasks.jsonl. No reference implementation or held-out assertion is included in runtime tasks. MBPP exposes {args.mbpp_examples} original assertion as an explicitly public example. HumanEval docstring examples remain in its original prompt.",
            "evaluation": "Freeze one complete Python solution per task before reading evaluation/tests.jsonl. Public benchmark tests are runtime-held-out, not secret or proven absent from pretraining.",
            "mbpp_entry_point": "Empty because the complete module can expose multiple public callables; signatures are in prompt.",
        },
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest["counts"]))


if __name__ == "__main__":
    main()
