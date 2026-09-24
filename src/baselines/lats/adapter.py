import ast
import dataclasses
import importlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
from threading import local
import time
import uuid

TASK_CONTEXT = local()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def mbpp_task(row):
    interface, assertion = row["prompt"].split("\n\nPublic example assertion:\n")
    description, interface = interface.split("\n\nImplement the following Python interface. Return a complete Python module.\n\n")
    module = ast.parse(interface)
    definitions = {node.name for node in module.body if isinstance(node, ast.FunctionDef)}
    parsed = ast.parse(assertion)
    assert len(parsed.body) == 1 and isinstance(parsed.body[0], ast.Assert)
    calls = {node.func.id for node in ast.walk(parsed) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    entry_points = calls & definitions
    assert len(entry_points) == 1
    entry_point = entry_points.pop()
    function = next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == entry_point)
    function.body.insert(0, ast.Expr(value=ast.Constant(value=description + '\n\nPublic example assertion.\n' + assertion)))
    return {"task_id": row["task_id"], "prompt": ast.unparse(ast.fix_missing_locations(module)), "entry_point": entry_point,
            "test": "def check(candidate):\n" + "\n".join("    " + line for line in assertion.splitlines())}


class ChatModel:
    is_chat = True

    def __init__(self, name, generate):
        self.name = name
        self.generate_messages = generate

    def generate_chat(self, messages, max_tokens=1024, temperature=0.2, num_comps=1):
        outputs = self.generate_messages(messages=[dataclasses.asdict(message) for message in messages],
                                         max_tokens=max_tokens, temperature=temperature, n=num_comps, stop=None)
        assert len(outputs) == num_comps
        return outputs[0] if num_comps == 1 else outputs


class SandboxExecutor:
    def __init__(self, executor_source, evaluator_path, sandbox, work_dir, memory_mb, result_type):
        self.executor_source = Path(executor_source).resolve(strict=True)
        self.sandbox_command = load_module("lats_sandbox_command", evaluator_path).sandbox_command
        self.sandbox = Path(sandbox).resolve(strict=True)
        self.work_dir = Path(work_dir).resolve()
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.memory_mb = memory_mb
        self.result_type = result_type
        self.evaluations = []
        self.calls = []
        self.dependencies = [(Path(importlib.util.find_spec("astunparse").origin).parent, "/dependencies/astunparse"),
                             (Path(importlib.util.find_spec("six").origin), "/dependencies/six.py")]

    def invoke(self, operation, arguments, timeout):
        started = time.monotonic()
        directory = Path(tempfile.mkdtemp(prefix="lats-tool-", dir=self.work_dir))
        program = directory / "check.py"
        marker = "LATS_RESULT_" + uuid.uuid4().hex
        limits = [("RLIMIT_AS", self.memory_mb * 1024**2), ("RLIMIT_CPU", math.ceil(timeout)),
                  ("RLIMIT_FSIZE", 1024**2), ("RLIMIT_NPROC", 32), ("RLIMIT_NOFILE", 64), ("RLIMIT_CORE", 0)]
        program.write_text(
            "import importlib\nimport json\nimport os\nimport resource\nimport sys\n"
            + "\n".join(f"resource.setrlimit(resource.{name}, ({value}, {value}))" for name, value in limits)
            + "\nsys.path[:0] = ['/upstream', '/dependencies']\nexecutor = importlib.import_module('executors.py_executor').PyExecutor()\n"
            + f"result = executor.{operation}(**{arguments!r})\n"
            + ("result = result._asdict()\n" if operation == "execute" else "")
            + f"print({marker!r} + json.dumps(result), flush=True)\nos._exit(0)\n")
        command = self.sandbox_command(self.sandbox, program)
        mounts = ["--dir", "/upstream", "--dir", "/dependencies", "--ro-bind",
                  str(self.executor_source), "/upstream/executors"]
        for source, destination in self.dependencies:
            mounts.extend(["--ro-bind", str(source), destination])
        position = command.index("--remount-ro")
        command[position:position] = mounts
        with (directory / "stdout").open("wb") as stdout, (directory / "stderr").open("wb") as stderr:
            process = subprocess.run(command, stdout=stdout, stderr=stderr, timeout=timeout + 2)
        lines = (directory / "stdout").read_text().splitlines()
        if process.returncode != 0 or not lines or not lines[-1].startswith(marker):
            raise RuntimeError(f"LATS executor did not complete, inspect {directory}")
        result = json.loads(lines[-1][len(marker):])
        self.calls.append({"operation": operation, "elapsed_seconds": time.monotonic() - started,
                           "arguments": arguments, "result": result, "directory": str(directory)})
        return result

    def execute(self, func, tests, timeout=5):
        if not tests:
            raise ValueError("LATS generated no valid internal tests")
        result = self.invoke("execute", {"func": func, "tests": tests, "timeout": timeout},
                             timeout * (1 + 2 * len(tests)))
        return self.result_type(result["is_passing"], result["feedback"], tuple(result["state"]))

    def evaluate(self, name, func, test, timeout=5):
        result = self.invoke("evaluate", {"name": name, "func": func, "test": test, "timeout": timeout}, timeout)
        self.evaluations.append({"solution": func, "public_test_passed": result})
        return result


def load_upstream(upstream):
    programming = str(Path(upstream).resolve() / "programming")
    sys.path.insert(0, programming)
    core = importlib.import_module("mcts")
    result_type = importlib.import_module("executors.executor_types").ExecuteResult
    assert Path(core.__file__).resolve() == Path(programming) / "mcts.py"
    assert Path(sys.modules['generators'].__file__).resolve().parent == Path(programming) / 'generators'
    core.model_factory = lambda name: TASK_CONTEXT.model
    core.executor_factory = lambda language, is_leet: TASK_CONTEXT.executor
    return core, result_type


def run_task(core, row, model, executor, *, max_iters, expansion_factor, number_of_tests, log_path, verbose):
    task = mbpp_task(row)
    TASK_CONTEXT.model = model
    TASK_CONTEXT.executor = executor
    core.run_mcts(dataset=[task], model_name=model.name, language="py", max_iters=max_iters, pass_at_k=1,
                  log_path=str(log_path), verbose=verbose, is_leetcode=False, n=expansion_factor,
                  number_of_tests=number_of_tests)
    return {"task_id": row["task_id"], "solution": executor.evaluations[-1]["solution"],
            "public_test_passed": executor.evaluations[-1]["public_test_passed"],
            "public_evaluations": executor.evaluations, "tool_calls": executor.calls}
