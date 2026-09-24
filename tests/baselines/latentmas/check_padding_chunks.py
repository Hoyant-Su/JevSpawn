import inspect
import json
from pathlib import Path

import torch

from baselines.latentmas.native_hybrid_transport import padding_segments
from transformers.models.qwen3_5.modeling_qwen3_5 import causal_conv1d_fn, causal_conv1d_update, torch_chunk_gated_delta_rule, torch_recurrent_gated_delta_rule


causal_conv1d_fn = inspect.unwrap(causal_conv1d_fn)
causal_conv1d_update = inspect.unwrap(causal_conv1d_update)
torch_chunk_gated_delta_rule = inspect.unwrap(torch_chunk_gated_delta_rule)
torch_recurrent_gated_delta_rule = inspect.unwrap(torch_recurrent_gated_delta_rule)

ROOT = Path(__file__).resolve().parents[3]
torch.set_num_threads(8)
torch.manual_seed(0)
rows = json.loads((ROOT / 'runs/latentmas-common-cpu-001/preflight.json').read_text())['records']
records = []
for begin in (0, 8):
    lengths = torch.tensor([row['role_tokens'][1] for row in rows[begin:begin + 8]])
    width, batch = int(lengths.max()), len(lengths)
    pads = width - lengths
    mask = torch.arange(width)[None, :] >= pads[:, None]
    boundaries = sorted({0, width, *pads.tolist()})
    chunks = padding_segments(mask)
    assert chunks == list(zip(boundaries, boundaries[1:]))
    assert all(bool((mask[:, start:end] == mask[:, start:start + 1]).all()) for start, end in chunks)
    channels, heads, dim, kernel = 12, 2, 4, 4
    hidden = torch.randn(batch, channels, width) * mask[:, None]
    weight = torch.randn(channels, kernel)
    initial_conv = torch.randn(batch, channels, kernel)
    token_conv, chunk_conv = initial_conv.clone(), initial_conv.clone()
    token_outputs, chunk_outputs = [], []
    for start in range(width):
        inactive = ~mask[:, start]
        saved = token_conv[inactive].clone()
        output = causal_conv1d_update(hidden[:, :, start:start + 1], token_conv, weight, activation='silu')
        token_conv[inactive] = saved
        token_outputs.append(output)
    for start, end in chunks:
        inactive = ~mask[:, start]
        saved = chunk_conv[inactive].clone()
        conv_history = torch.cat([chunk_conv, hidden[:, :, start:end]], dim=-1)
        chunk_conv.copy_(conv_history[:, :, -kernel:])
        output = causal_conv1d_fn(conv_history, weight, activation='silu')[:, :, -(end - start):]
        chunk_conv[inactive] = saved
        assert torch.equal(chunk_conv[inactive], saved)
        chunk_outputs.append(output)
    token_output = torch.cat(token_outputs, dim=-1)
    chunk_output = torch.cat(chunk_outputs, dim=-1)
    conv_error = float(((token_output - chunk_output) * mask[:, None]).abs().max())
    torch.testing.assert_close(token_conv, chunk_conv)
    torch.testing.assert_close(token_output * mask[:, None], chunk_output * mask[:, None])
    q, k, v = [torch.randn(batch, width, heads, dim) for _ in range(3)]
    g = -torch.rand(batch, width, heads)
    beta = torch.rand(batch, width, heads)
    initial_state = torch.randn(batch, heads, dim, dim)
    token_state, chunk_state = initial_state.clone(), initial_state.clone()
    token_outputs, chunk_outputs = [], []
    for start in range(width):
        inactive = ~mask[:, start]
        saved = token_state[inactive].clone()
        output, token_state = torch_recurrent_gated_delta_rule(
            q[:, start:start + 1], k[:, start:start + 1], v[:, start:start + 1],
            g[:, start:start + 1], beta[:, start:start + 1], initial_state=token_state,
            output_final_state=True, use_qk_l2norm_in_kernel=True)
        token_state[inactive] = saved
        token_outputs.append(output)
    for start, end in chunks:
        inactive = ~mask[:, start]
        saved = chunk_state[inactive].clone()
        output, chunk_state = torch_chunk_gated_delta_rule(
            q[:, start:end], k[:, start:end], v[:, start:end], g[:, start:end], beta[:, start:end],
            initial_state=chunk_state, output_final_state=True, use_qk_l2norm_in_kernel=True)
        chunk_state[inactive] = saved
        assert torch.equal(chunk_state[inactive], saved)
        chunk_outputs.append(output)
    token_output = torch.cat(token_outputs, dim=1)
    chunk_output = torch.cat(chunk_outputs, dim=1)
    recurrent_error = float(((token_output - chunk_output) * mask[:, :, None, None]).abs().max())
    torch.testing.assert_close(token_state, chunk_state, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(token_output * mask[:, :, None, None], chunk_output * mask[:, :, None, None], rtol=1e-5, atol=1e-6)
    records.append({'batch_size': batch, 'physical_prompt_width': width, 'valid_tokens': int(mask.sum()),
                    'padding_lengths': pads.tolist(), 'boundaries': boundaries,
                    'original_appended_role_forwards': int(pads.max()) + 1,
                    'segmented_appended_role_forwards': len(chunks),
                    'convolution_valid_output_max_absolute_error': conv_error,
                    'recurrent_valid_output_max_absolute_error': recurrent_error,
                    'recurrent_final_state_max_absolute_error': float((token_state - chunk_state).abs().max()),
                    'inactive_states_restored_exactly': True})
result = {'scope': 'CPU operator unit evidence using installed Qwen3.5 PyTorch reference convolution and delta-rule functions (explicitly unwrapped from CUDA dispatch); synthetic operator tensors and real qualification padding widths. Not model inference or accuracy evidence.',
          'operator_fixture': {'dtype': 'float32', 'seed': 0, 'channels': channels, 'heads': heads, 'dim': dim, 'kernel': kernel},
          'records': records}
(ROOT / 'runs/latentmas-padding-cpu-proof-001/proof.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result))
