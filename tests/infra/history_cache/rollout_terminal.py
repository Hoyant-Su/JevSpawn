import argparse
import ast
from functools import partial
import inspect
import json
from pathlib import Path
import textwrap
from types import FunctionType

import torch.distributed as dist

from baselines.common.config import SharedConfig
from baselines.common.parallel_run import initialize_parallel
from baselines.common.persistence import save
from baselines.common.parallel_service import ParallelService
from baselines.common.run import run_with_environment
from baselines.common.runtime import InferenceRuntime
from baselines.common.service_factory import service_factory
from jev_spawn.infra.configuration import resolve_symbol


def test_runtime(source):
    def parallel_factory(factory):
        constructor = factory.func
        service = type('Parallel' + constructor.__name__, (ParallelService, constructor),
                       {'parallel_source_service': source})
        return partial(service, *factory.args, **factory.keywords)

    original = InferenceRuntime.__init__
    initialize = FunctionType(original.__code__, dict(original.__globals__, parallel_factory=parallel_factory),
                              original.__name__, original.__defaults__, original.__closure__)
    return type('TestRuntime', (InferenceRuntime,), {'__init__': initialize})


def local_runner(inference_path, runtime):
    tree = ast.parse(textwrap.dedent(inspect.getsource(run_with_environment)))
    function, = tree.body
    assignment, = [node for node in function.body if isinstance(node, ast.Assign)
                   and ast.unparse(node.targets[0]) == 'inference']
    assignment.value = ast.parse('read(inference_path)', mode='eval').body
    ast.fix_missing_locations(tree)
    namespace = dict(run_with_environment.__globals__, inference_path=inference_path, InferenceRuntime=runtime)
    exec(compile(tree, inspect.getsourcefile(run_with_environment), 'exec'), namespace)
    return namespace[function.name]


def run(settings):
    specification = json.loads(Path(settings['specification']).read_text())
    assert specification['shared_config'] == settings['shared_config']
    shared = SharedConfig.load(settings['shared_config'])
    backend, commands, starts = initialize_parallel(shared, json.loads(Path(settings['parallel_settings']).read_text()))
    inference = json.loads(Path(settings['inference_settings']).read_text())
    method = json.loads(Path(specification['method']).read_text())
    runtime_class = test_runtime(settings['parallel_source_service'])
    output = Path(settings['run_output'])
    if commands.is_leader:
        output.mkdir(parents=True, exist_ok=True)
        save(output / 'parallel_startup.json', {'ranks': starts, 'test_configuration': settings})
        definition = specification['environment_execution']
        factory = partial(resolve_symbol(definition['class']), **definition['parameters'])
        try:
            local_runner(settings['inference_settings'], runtime_class)(specification, output, backend,
                environment_factory=factory, environment_contract={'environment_execution': definition})
        finally:
            commands.finish()
    else:
        runtime = runtime_class(specification['shared_config'], service_factory(method, inference), backend=backend)
        try:
            commands.serve()
        finally:
            runtime.close()
    dist.destroy_process_group(commands.control_group)
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    run(json.loads(parser.parse_args().config.read_text()))
