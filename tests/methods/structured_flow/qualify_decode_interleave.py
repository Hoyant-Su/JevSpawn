import argparse
from concurrent.futures import Future
from copy import deepcopy
import json
from pathlib import Path
import time

import torch

from baselines.common.config import SharedConfig
from baselines.common.deadlines import SampleDeadlines
from baselines.common.service import BatchedRequest, BatchService
from jev_spawn.infra.backend import Backend
from jev_spawn.schema import CONTROLLER
from methods.program_execution.grouped import score_grouped
from tests.methods.structured_flow.qualify_prefix_cache import native_state, workload


@torch.inference_mode()
def qualify(config, output):
    shared = SharedConfig.load(config['shared_config'])
    backend = Backend(shared.backend())
    source = json.loads(Path(config['autoregressive_batches']).read_text())[config['batch_index']]
    finite = json.loads(Path(config['finite_workload']).read_text())
    CONTROLLER['option_template'] = json.loads(Path(finite['method_prompts']).read_text())['option_template']
    task, groups = workload(finite)
    expected = [score_grouped(backend, [group], config['field_mode'])['groups'] for group in groups]
    records, tensors_checked = [], []
    for mode in config['conditions']:
        deadlines = SampleDeadlines(shared.runtime.sample_timeout_seconds)
        for task_id in dict.fromkeys(source['task_ids']):
            deadlines.start(task_id)
        service = BatchService(backend, shared, deadlines)
        service.close()
        batch = [BatchedRequest(messages, budget, source['temperature'], tuple(source['row_stops'][i]),
                    task_id, time.perf_counter(), Future())
                 for i, (messages, budget, task_id) in enumerate(zip(source['messages'],
                     source['requested_max_new_tokens'], source['task_ids']))]
        valid = service._validate_inputs(batch)
        assert len(valid) == len(batch)
        actual = []

        def boundary():
            if len(actual) == len(groups):
                return
            decoder = next(iter(service.decoders.values()))
            tensors, metadata = native_state(decoder.cache)
            before = {name: tensor.clone() for name, tensor in tensors.items()}
            ids, positions, logits = decoder.ids.clone(), decoder.positions.clone(), decoder.logits.clone()
            service.interleaving_finite = True
            try:
                actual.append(score_grouped(backend, [groups[len(actual)]], config['field_mode'])['groups'])
            finally:
                service.interleaving_finite = False
            after, after_metadata = native_state(decoder.cache)
            assert metadata == after_metadata
            assert all(torch.equal(before[name], after[name]) for name in before)
            assert torch.equal(ids, decoder.ids) and torch.equal(positions, decoder.positions) and torch.equal(logits, decoder.logits)
            tensors_checked.append(len(before))

        if mode == 'decode_step':
            service._between_decode_steps = boundary
        started = time.perf_counter()
        service._generate(valid)
        record = deepcopy(service.records[-1])
        if mode == 'decode_step':
            assert actual == expected, 'Finite choices/logits/probabilities differ.'
        records.append({'mode': mode, 'wall_seconds': time.perf_counter() - started,
                        'batch': record, 'finite_groups': actual})
        del service
        torch.cuda.empty_cache()
    assert records[0]['batch']['output_token_ids'] == records[1]['batch']['output_token_ids'], 'AR tokens changed.'
    result = {'configuration': config, 'autoregressive_tokens_exact': True, 'finite_scores_exact': True,
              'native_attention_conv_recurrent_unchanged': True, 'audited_tensor_counts': tensors_checked,
              'scope': 'Paired replay of one original eight-row AR batch and ten original finite fields; no new workers or benchmark quality claim.',
              'records': records}
    output.mkdir(parents=True, exist_ok=True)
    (output / 'completion.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: value for key, value in result.items() if key not in ['records', 'configuration']}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    qualify(json.loads(args.config.read_text()), args.output)


if __name__ == '__main__':
    main()
