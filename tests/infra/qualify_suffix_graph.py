import argparse
from concurrent.futures import Future
from contextlib import nullcontext
from functools import partial
import json
from pathlib import Path
import time

import torch
import torch.distributed as dist

from baselines.common.config import SharedConfig
from baselines.common.parallel_run import initialize_parallel
from baselines.common.parallel_service import decode_request, encode_request
from baselines.common.runtime import InferenceRuntime
from baselines.common.shared_state_prefix_service import SharedStatePrefixService
from jev_spawn.infra.prompts import load_prompt
from jev_spawn.schema import CONTROLLER, controller_prompts
from suffix_graph import SuffixGraphs


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')




def run(settings):
    shared = SharedConfig.load(settings['shared_config'])
    parallel = json.loads(Path(settings['parallel_settings']).read_text())
    backend, commands, startup = initialize_parallel(shared, parallel)
    source = json.loads(Path(settings['source_protocol']).read_text())
    method = json.loads(Path(settings['method']).read_text())
    runtime = InferenceRuntime(settings['shared_config'], partial(SharedStatePrefixService,
        settings=method['settings'], prompts=load_prompt(method['prompts'])), backend=backend)
    CONTROLLER.clear()
    CONTROLLER.update(source['controller'])
    output = Path(settings['output'])
    if commands.is_leader:
        output.mkdir(parents=True, exist_ok=False)
        save(output / 'protocol.json', {'settings': settings, 'startup': startup, 'source_groups': source['groups']})
    dist.barrier()
    records = []
    graphs = SuffixGraphs(backend, settings['graph'])

    @torch.inference_mode()
    def score(payload):
        batch = [decode_request(value) for value in payload['requests']]
        graph_mode = payload['phase'] in settings['graph_phases']
        instrumentation = graphs.installed() if graph_mode else nullcontext()
        torch.cuda.synchronize(backend.device)
        started = time.perf_counter()
        with instrumentation:
            result = runtime.service._score(batch)
            torch.cuda.synchronize(backend.device)
        local_wall = time.perf_counter() - started
        metrics = torch.tensor([local_wall, *result['timings'].values()], device=backend.device, dtype=torch.float64)
        dist.all_reduce(metrics, op=dist.ReduceOp.MAX)
        values = metrics.tolist()
        reference = json.loads((Path(settings['reference_run']) /
            f"measured-admitted-{payload['batch_index']}.json").read_text())
        expected = [node for group in reference['result']['groups'] for node in group]
        actual = [node for group in result['groups'] for node in group]
        assert [(node['id'], node['input_tokens'], node['option_ids']) for node in actual] == [
            (node['id'], node['input_tokens'], node['option_ids']) for node in expected]
        exact = result['groups'] == reference['result']['groups']
        max_error = max(abs(a-b) for x, y in zip(actual, expected)
                        for a, b in zip(x['option_logits'], y['option_logits']))
        record = {'phase': payload['phase'], 'batch_index': payload['batch_index'],
            'rank': dist.get_rank(), 'local_scoring_seconds': local_wall, 'rankmax_scoring_seconds': values[0],
            'rankmax_phase_seconds': dict(zip(result['timings'], values[1:])), 'result': result,
            'exact_original_logits_choices': exact, 'max_logit_absolute_error': max_error,
            'choice_mismatches': sum(x['choice'] != y['choice'] for x,y in zip(actual,expected)),
            'graph_count': len(graphs.graphs),
            'capture_seconds': [graph.capture_seconds for graph in graphs.graphs.values()],
            'allocated_bytes': torch.cuda.memory_allocated(backend.device),
            'reserved_bytes': torch.cuda.memory_reserved(backend.device)}
        save(output / f"rank{dist.get_rank()}-{payload['phase']}-batch{payload['batch_index']}.json", record)
        records.append(record)
        return record

    commands.register(settings['command'], score)
    if commands.is_leader:
        identity = json.loads(Path(settings['source_settings']).read_text())['task_id']
        runtime.deadlines.start(identity)
        try:
            for phase in settings['phases']:
                for index, group in enumerate(source['groups'], start=1):
                    started = time.perf_counter()
                    batch = []
                    for node in group:
                        prompt = controller_prompts([node['state']], node['question'], node['options'],
                            list(backend.answer_labels[:len(node['options'])]), CONTROLLER['output_instruction'])[0]
                        batch.append(runtime.service.request_type(
                            [{'role': 'system', 'content': CONTROLLER['system']}, {'role': 'user', 'content': prompt}],
                            settings['decision_tokens'], shared.generation.temperature, (), identity,
                            time.perf_counter(), Future(), field=node))
                    batch = runtime.service._validate_inputs(batch)
                    assert len(batch) == len(group)
                    admission = time.perf_counter() - started
                    record = commands.call(settings['command'], {'phase': phase, 'batch_index': index,
                        'requests': [encode_request(request) for request in batch]})
                    record.update(admission_seconds=admission,
                                  inclusive_seconds=time.perf_counter() - started)
                    save(output / f'{phase}-batch{index}.json', record)
            runtime.close()
        finally:
            commands.finish()
    else:
        commands.serve()
        runtime.close()
    graphs.graphs.clear()
    if commands.is_leader:
        save(output / 'completion.json', {'settings': settings, 'records': records})
        print(json.dumps({'completed_cohorts': len(records), 'all_original_logits_choices_exact': all(x['exact_original_logits_choices'] for x in records)}), flush=True)
    dist.destroy_process_group(commands.control_group)
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.config.read_text()))
