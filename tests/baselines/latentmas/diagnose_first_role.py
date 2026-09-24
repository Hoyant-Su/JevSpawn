import argparse
import json
from pathlib import Path

import torch

from baselines.latentmas.adapter import HybridTransport, ModelAdapter, method, task_messages
from baselines.latentmas.adapter import task_item
from baselines.latentmas.numerics import compare
from jev_spawn.infra.backend import Backend


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')



class ObservedTransport(HybridTransport):
    def __init__(self, backend):
        super().__init__(backend)
        self.frames = []
        self.observed = None
        self.hook = self.trunk.register_forward_pre_hook(self.observe_inputs, with_kwargs=True)

    def observe_inputs(self, module, args, kwargs):
        width = kwargs['inputs_embeds'].shape[1]
        self.observed = {
            'positions': kwargs['position_ids'].detach().cpu().clone(),
            'query_mask': kwargs['attention_mask'][:, -width:].detach().cpu().clone(),
            'input_last': kwargs['inputs_embeds'][:, -1].detach().cpu().clone(),
        }

    def _forward(self, embeddings, mask, cache):
        output = super()._forward(embeddings, mask, cache)
        frame = self.snapshot()
        frame.update(self.observed)
        frame['physical_cache_length'] = int(self.cache.get_seq_length())
        frame['cache_classes'] = [type(layer).__name__ for layer in self.cache.layers]
        assert torch.equal(output.hidden_states[-1][:, -1].detach().cpu(), frame['hidden'])
        self.frames.append(frame)
        return output


class ObservedAdapter(ModelAdapter):
    def __init__(self, backend, args):
        super().__init__(backend, args)
        self.model = ObservedTransport(backend)
        self.alignment_frames = []

    def _apply_latent_realignment(self, hidden, model):
        aligned = super()._apply_latent_realignment(hidden, model)
        self.alignment_frames.append({
            'hidden_input': hidden.detach().cpu().clone(),
            'pre_aligned': self.pre_aligned.detach().cpu().clone(),
            'aligned_output': aligned.detach().cpu().clone(),
        })
        return aligned

    def reset(self, capture=False):
        super().reset(capture=capture)
        self.model.frames = []
        self.alignment_frames = []


def trajectory(wrapper, input_ids, mask, latent_steps):
    wrapper.reset()
    assert bool(mask[:, -1].all())
    wrapper.generate_latent_batch(input_ids, mask, latent_steps=latent_steps, past_key_values=None)
    assert len(wrapper.model.frames) == latent_steps + 1
    assert len(wrapper.alignment_frames) == latent_steps
    return {'states': wrapper.model.frames, 'alignments': wrapper.alignment_frames,
            'forward_records': wrapper.model.records}



def hidden_payload(trace):
    return {'states': [{key: frame[key] for key in ['mask', 'positions', 'hidden', 'input_last']}
                       for frame in trace['states']], 'alignments': trace['alignments']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    original = json.loads(Path(config['method_config']).read_text())
    native = json.loads(Path(original['native_config']).read_text())
    tasks = [json.loads(line) for line in Path(original['tasks']).read_text().splitlines()]
    tasks = tasks[:original['task_count']]
    assert len(tasks) == native['batch_size'] == original['root_batch_size'] == 8
    assert original['latent_steps'] == config['latent_steps'] == 10
    assert config['comparisons'] == ['unpadded_b1', 'padded_width_b1']
    args.output.mkdir(parents=True, exist_ok=False)
    save(args.output / 'protocol.json', {'diagnostic': config, 'method': original, 'native': native,
                                       'task_ids': [t['task_id'] for t in tasks]})
    backend = Backend(native)
    runner = method(backend, original, tasks)
    wrapper = ObservedAdapter(backend, runner.args)
    messages = [task_messages(role=runner.agents[0].role, question=task_item(t)['question'],
                             context='', method=runner.method_name, args=runner.args) for t in tasks]
    prompts, ids, mask, tokens = wrapper.prepare_chat_batch(messages)
    save(args.output / 'inputs.json', {'prompts': prompts, 'input_ids': ids.tolist(),
                                      'attention_mask': mask.tolist(), 'valid_lengths': mask.sum(1).tolist()})
    reports = {name: [] for name in config['comparisons']}
    try:
        with torch.inference_mode():
            matrix, target = wrapper._ensure_latent_realign_matrix(wrapper.model, backend.device, runner.args)
            save(args.output / 'setup.json', {'backend': {k: v for k, v in backend.metadata.items() if k != 'controller'},
                 'alignment_shape': list(matrix.shape),
                 'alignment_dtype': str(matrix.dtype), 'alignment_frobenius_norm': float(matrix.norm()),
                 'target_embedding_norm': float(target), 'first_role': runner.agents[0].role})
            batched = trajectory(wrapper, ids, mask, original['latent_steps'])
            torch.save(hidden_payload(batched), args.output / 'batched_hidden.pt')
            save(args.output / 'batched_forward_records.json', batched['forward_records'])
            for comparison in config['comparisons']:
                for row, task in enumerate(tasks):
                    selected_ids, selected_mask = ids[row:row + 1], mask[row:row + 1]
                    if comparison == 'unpadded_b1':
                        _, selected_ids, selected_mask, _ = wrapper.prepare_chat_batch([messages[row]])
                    assert torch.equal(selected_ids[0][selected_mask[0].bool()], ids[row][mask[row].bool()])
                    single = trajectory(wrapper, selected_ids, selected_mask, original['latent_steps'])
                    report = {'task_id': task['task_id'], 'comparison': comparison,
                              'valid_input_ids_equal': True,
                              **compare(batched, single, row, comparison == 'padded_width_b1')}
                    filename = comparison + '-' + task['task_id'].replace('/', '_')
                    save(args.output / (filename + '.json'), report)
                    torch.save(hidden_payload(single), args.output / (filename + '-hidden.pt'))
                    reports[comparison].append({'task_id': task['task_id'],
                        'hidden_relative_l2': [s['hidden']['relative_l2'] for s in report['states']],
                        'hidden_max_absolute': [s['hidden']['max_absolute'] for s in report['states']],
                        'aligned_input_relative_l2': [s['aligned_output']['relative_l2'] for s in report['alignments']]})
                    save(args.output / 'summary.json', {'scope': 'First role numerical diagnostic. No answer decoding or quality evaluation.',
                                                        'comparisons': reports})
                    print(json.dumps({'comparison': comparison, 'task_id': task['task_id'],
                                      'hidden_relative_l2': reports[comparison][-1]['hidden_relative_l2']}), flush=True)
    finally:
        wrapper.model.hook.remove()


if __name__ == '__main__':
    main()
