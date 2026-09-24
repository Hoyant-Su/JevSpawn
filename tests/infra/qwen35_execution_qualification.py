import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from baselines.common.config import SharedConfig
from jev_spawn.infra.backend import Backend
from jev_spawn.infra.qwen35 import execution
from jev_spawn.infra.qwen35.gates import gdn_gates
from jev_spawn.runtime.decoding import CapturedDecode
from jev_spawn.schema import CONTROLLER
from qualify_finite_prefix_independent import render, workload


def compare(actual, expected, settings):
    difference = (actual.float() - expected.float()).abs()
    return {'exact': torch.equal(actual, expected), 'max_absolute_error': difference.max().item(),
            'mean_absolute_error': difference.mean().item(),
            'within_tolerance': torch.allclose(actual.float(), expected.float(),
                atol=settings['absolute_tolerance'], rtol=settings['relative_tolerance'])}


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def tokens(decoder, inputs, steps, replay):
    result = [decoder.prefill(inputs)]
    for _ in range(steps):
        decoder.graph.replay() if replay else decoder.step()
        result.append(decoder.ids.squeeze(-1).clone())
    return torch.stack(result, dim=-1)


@torch.inference_mode()
def run(settings):
    output = Path(settings['output'])
    output.mkdir(parents=True, exist_ok=False)
    replay = json.loads(Path(settings['replay_specification']).read_text())
    protocol, context, groups, recorded = workload(replay)
    CONTROLLER.clear()
    CONTROLLER.update(replay['controller_snapshot'])
    CONTROLLER['option_template'] = protocol['prompts']['option_template']
    shared = SharedConfig.load(settings['shared_config'])
    backend = Backend(shared.backend())
    selected = [groups[group][node] for group, node in settings['requests']]
    rendered = render(backend, selected, context)
    actual_lengths = list(map(len, rendered['input_ids']))
    expected_lengths = [recorded[group][node]['input_tokens'] for group, node in settings['requests']]
    assert actual_lengths == expected_lengths
    inputs, _ = backend._encode(rendered['rendered'])
    assert inputs['input_ids'].shape[-1] == max(expected_lengths)
    assert inputs['attention_mask'].sum(-1).tolist() == expected_lengths
    install_settings = json.loads(Path(settings['execution_settings']).read_text())
    save(output / 'protocol.json', {'settings': settings, 'execution_settings': install_settings,
         'requests': selected, 'input_ids': rendered['input_ids'], 'batch_size': len(selected),
         'input_lengths': actual_lengths, 'mask': inputs['attention_mask'].tolist(),
         'scope': 'Paired real recorded requests, complete inputs, fixed decode steps for numerical qualification.'})
    language = backend.model.get_submodule(install_settings['language_model_path'])
    mixer = language.get_submodule(settings['gdn_module'])
    captured, handles = {}, []
    def capture(name):
        def hook(module, args, result):
            captured[name] = {'input': tuple(value.clone() for value in args), 'output': result.clone()}
        return hook
    for name in install_settings['gdn_projections']:
        handles.append(getattr(mixer, name).register_forward_hook(capture(name)))
    handles.append(mixer.norm.register_forward_hook(capture('gated_norm')))
    rms = language.get_submodule(settings['rms_module'])
    handles.append(rms.register_forward_hook(capture('rms_norm')))
    capacity = inputs['input_ids'].shape[-1] + settings['decode_steps'] + shared.runtime.graph_warmup_steps
    native = CapturedDecode(backend, len(selected), capacity)
    native.prefill(inputs)
    for handle in handles:
        handle.remove()
    native_logits = native.logits.clone()
    native_tokens = tokens(native, inputs, settings['decode_steps'], False)
    del native
    torch.cuda.empty_cache()
    packing, original_pack = [], execution.pack_input_projections
    def checked_pack(module, names):
        weights = torch.cat([getattr(module, name).weight.detach() for name in names])
        original_pack(module, names)
        exact = torch.equal(weights, module._input_projection_weight)
        assert exact
        packing.append({'module': type(module).__name__, 'projections': names,
                        'shape': list(weights.shape), 'exact_weights': exact,
                        'shared_storage': all(getattr(module, name).weight.untyped_storage().data_ptr()
                                              == module._input_projection_weight.untyped_storage().data_ptr()
                                              for name in names)})
    execution.pack_input_projections = checked_pack
    execution.install_qwen35_execution(backend.model, install_settings)
    execution.pack_input_projections = original_pack
    backend.config['execution'] = settings['optimized_execution']
    operations = {}
    projection_input = captured[settings['projection_input']]['input'][settings['input_index']]
    projected = mixer.input_projections(projection_input)
    for name, actual in zip(install_settings['gdn_projections'], projected, strict=True):
        module = getattr(mixer, name)
        fp32 = F.linear(projection_input.float(), module.weight.float())
        operations[name] = {'native': compare(actual, captured[name]['output'], settings['tolerance']),
                            'optimized_fp32': compare(actual, fp32, settings['tolerance']),
                            'native_fp32': compare(captured[name]['output'], fp32, settings['tolerance'])}
    a, b = captured['in_proj_a']['output'], captured['in_proj_b']['output']
    g, beta = gdn_gates(mixer, a, b)
    operations['gdn_decay_gate'] = compare(g, -mixer.A_log.float().exp() *
                                         F.softplus(a.float() + mixer.dt_bias), settings['gate_tolerance'])
    operations['gdn_beta'] = compare(beta, b.sigmoid(), settings['tolerance'])
    operations['gated_norm'] = compare(mixer.norm(*captured['gated_norm']['input']),
                                        captured['gated_norm']['output'], settings['tolerance'])
    operations['rms_norm'] = compare(rms(*captured['rms_norm']['input']),
                                      captured['rms_norm']['output'], settings['tolerance'])
    save(output / 'operations.json', {'packing': packing, 'operations': operations})
    captured.clear()
    optimized = CapturedDecode(backend, len(selected), capacity)
    optimized.prefill(inputs)
    optimized_logits = optimized.logits.clone()
    labels = [torch.tensor(backend.answer_label_ids[:len(request['options'])], device=backend.device)
              for request in selected]
    finite = [{'node': request['id'],
               'native': request['options'][int(native_logits[index, ids].argmax())]['id'],
               'optimized': request['options'][int(optimized_logits[index, ids].argmax())]['id']}
              for index, (request, ids) in enumerate(zip(selected, labels, strict=True))]
    eager_tokens = tokens(optimized, inputs, settings['decode_steps'], False)
    optimized.prefill(inputs)
    optimized.capture(shared.runtime.graph_warmup_steps)
    graph_tokens = tokens(optimized, inputs, settings['decode_steps'], True)
    report = {'finite_decisions': finite, 'native_tokens': native_tokens.tolist(),
              'optimized_eager_tokens': eager_tokens.tolist(), 'optimized_graph_tokens': graph_tokens.tolist(),
              'native_vs_optimized_tokens_equal': torch.equal(native_tokens, eager_tokens),
              'eager_vs_graph_tokens_equal': torch.equal(eager_tokens, graph_tokens),
              'full_logits': compare(optimized_logits, native_logits, settings['logit_tolerance']),
              'all_weights_exact': all(row['exact_weights'] and row['shared_storage'] for row in packing),
              'finite_choices_equal': all(row['native'] == row['optimized'] for row in finite),
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'not_a_task_accuracy_or_throughput_experiment': True}
    save(output / 'completion.json', report)
    print(json.dumps(report), flush=True)
    assert report['all_weights_exact'] and report['eager_vs_graph_tokens_equal']


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--specification', type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.specification.read_text()))
