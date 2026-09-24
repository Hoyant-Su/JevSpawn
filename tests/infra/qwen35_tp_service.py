import argparse
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import json
from pathlib import Path
from threading import Barrier, Event
import time

from safetensors import safe_open
import torch
import torch.distributed as dist

from baselines.common.config import SharedConfig
from baselines.common.parallel_run import initialize_parallel
from baselines.common.runtime import InferenceRuntime
from baselines.common.shared_state_prefix_service import SharedStatePrefixService
from jev_spawn.infra.prompts import load_prompt
from jev_spawn.schema import CONTROLLER
from qualify_shared_state_prefix import workload


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


@torch.inference_mode()
def check_readout(backend, settings):
    head = backend.model.lm_head
    ids = tuple(dict.fromkeys([*backend.answer_label_ids,
        *range(0, head.vocab_size, head.weight.shape[0]), head.vocab_size - 1]))
    selected = backend.selected_output_weights(ids)
    model_path = Path(backend.config['model_path'])
    index = json.loads((model_path / settings['checkpoint_index']).read_text())
    shard = model_path / index['weight_map'][settings['oracle_weight_name']]
    with safe_open(shard, framework='pt', device=backend.device.index) as archive:
        original = archive.get_tensor(settings['oracle_weight_name'])
    oracle = original.index_select(0, torch.tensor(ids, device=backend.device))
    exact = torch.equal(selected, oracle)
    assert exact
    result = {'token_ids': ids, 'shape': list(selected.shape), 'exact_original_checkpoint_rows': exact,
              'source': str(shard), 'source_tensor': settings['oracle_weight_name']}
    del original, oracle
    torch.cuda.empty_cache()
    return result


def leader_work(runtime, backend, settings, payload):
    source, fields = payload['text'], payload['fields']
    service, deadlines = runtime.service, runtime.deadlines
    for identity in [*source['task_ids'], payload['finite_task_id']]:
        deadlines.start(identity)
    barrier, decoding = Barrier(len(source['task_ids'])), Event()
    between = service._between_decode_steps

    def signal_decode():
        decoding.set()
        return between()

    service._between_decode_steps = signal_decode

    def text_request(row):
        barrier.wait()
        result, = service.complete(source['messages'][row], settings['text_max_new_tokens'],
            runtime.config.generation.temperature, stop=source['stop'],
            task_id=source['task_ids'][row], return_tokens=True)
        return {'task_id': source['task_ids'][row], **result}

    def finite_requests():
        decoding.wait()
        return service.decide(fields, task_id=payload['finite_task_id'])

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=settings['worker_threads']) as workers:
        finite = workers.submit(finite_requests)
        texts = [workers.submit(text_request, row) for row in range(len(source['task_ids']))]
        text_results = [future.result() for future in texts]
        finite_results = finite.result()
    interleaved_seconds = time.perf_counter() - started
    service._between_decode_steps = between
    stop = service._stopping
    identity = settings['deadline_identity']
    deadlines.start(identity)

    def bounded_deadline(batch, width):
        assert [request.task_id for request in batch] == [identity]
        deadlines.started[identity] = (time.perf_counter() - deadlines.seconds
                                       + settings['deadline_window_seconds'])
        return stop(batch, width)

    service._stopping = bounded_deadline
    expired = False
    try:
        service.complete(source['messages'][settings['deadline_source_row']],
            runtime.config.generation.max_new_tokens, runtime.config.generation.temperature,
            stop=source['stop'], task_id=identity, return_tokens=True)
    except TimeoutError:
        expired = True
    assert expired
    runtime.close()
    service._stopping = stop
    records = service.records
    finite_batches = [record for record in records if record.get('operation') == 'finite']
    text_batches = [record for record in records if record.get('operation') != 'finite'
                    and identity not in record['task_ids']]
    timeout_batches = [record for record in records if identity in record['task_ids']]
    assert sum(record['batch_size'] for record in finite_batches) == len(fields)
    assert any(record.get('interleaved_at') == 'decode_step' for record in finite_batches)
    assert len(text_batches) == 1 and text_batches[0]['batch_size'] == len(source['task_ids'])
    assert len(timeout_batches) == 1 and timeout_batches[0]['finish_reasons'] == ['timeout']
    assert dict(zip(text_batches[0]['task_ids'], text_batches[0]['input_tokens'])) == dict(
        zip(source['task_ids'], source['input_tokens']))
    return {'interleaved_seconds': interleaved_seconds, 'text_results': text_results,
            'finite_results': finite_results, 'deadline_exception_observed': expired,
            'nested_finite_batches': sum(record.get('interleaved_at') == 'decode_step' for record in finite_batches),
            'batches': records, 'metadata': runtime.metadata(), 'input_failures': service.input_failures}


def run(settings):
    shared = SharedConfig.load(settings['shared_config'])
    parallel = json.loads(Path(settings['parallel_settings']).read_text())
    backend, commands, startup = initialize_parallel(shared, parallel)
    oracle = check_readout(backend, settings)
    output = Path(settings['output'])
    payload = None
    if commands.is_leader:
        output.mkdir(parents=True, exist_ok=False)
        finite_settings = json.loads(Path(settings['finite_source']).read_text())
        protocol, context, groups, originals, calls = workload(finite_settings)
        fields = [{'id': node['id'], 'context': context, 'input': calls[node['id']]['input'],
                   'question': calls[node['id']]['question'], 'options': calls[node['id']]['options']}
                  for group in groups for node in group]
        payload = {'text': json.loads(Path(settings['text_source']).read_text())[settings['text_batch_index']],
                   'fields': fields, 'finite_task_id': finite_settings['task_id']}
        save(output / 'protocol.json', {'settings': settings, 'startup': startup, 'readout_oracle': oracle,
            'inputs': payload, 'deadline_semantics': 'Leader deadline shortened at actual numerical dispatch; synchronized stop state governs all ranks.'})
    payload = commands.leader_value(payload)
    CONTROLLER.clear()
    CONTROLLER.update(load_prompt(settings['controller_prompt']))
    method = json.loads(Path(settings['method']).read_text())
    runtime = InferenceRuntime(settings['shared_config'], partial(SharedStatePrefixService,
        settings=method['settings'], prompts=load_prompt(method['prompts'])), backend=backend)
    dist.barrier()
    if commands.is_leader:
        try:
            result = leader_work(runtime, backend, settings, payload)
            save(output / 'batches.json', result['batches'])
            save(output / 'completion.json', result)
            print(json.dumps({key: result[key] for key in ('interleaved_seconds', 'nested_finite_batches',
                                                          'deadline_exception_observed')}), flush=True)
        finally:
            commands.finish()
    else:
        commands.serve()
        runtime.close()
    rank_records = {'rank': dist.get_rank(), 'batches': [{key: record[key] for key in
        ('task_ids', 'batch_size', 'finish_reasons', 'output_token_ids') if key in record}
        for record in runtime.service.records]}
    all_records = [None] * shared.runtime.world_size
    dist.all_gather_object(all_records, rank_records, group=commands.control_group)
    if commands.is_leader:
        save(output / 'rank_batches.json', all_records)
        reference = all_records[parallel['leader_rank']]['batches']
        assert all(record['batches'] == reference for record in all_records)
    dist.destroy_process_group(commands.control_group)
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.config.read_text()))
