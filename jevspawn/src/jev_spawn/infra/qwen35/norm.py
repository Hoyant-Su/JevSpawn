import torch
import triton

from jev_spawn.infra.fused_norm import rms_kernel


def normalize(module, hidden_states, gate):
    width = hidden_states.shape[-1]
    values = hidden_states.reshape(-1, width)
    gates = values if gate is None else gate.reshape(-1, width)
    output = torch.empty(hidden_states.shape, dtype=hidden_states.dtype, device=hidden_states.device)
    epsilon = module.eps if gate is None else module.variance_epsilon
    rms_kernel[(values.shape[0],)](
        values, module.weight, gates, output, values.stride(0), gates.stride(0), width,
        epsilon, gate is not None, triton.next_power_of_2(width), **module._norm_launch)
    return output


def normalized(module, hidden_states):
    return normalize(module, hidden_states, None)


def gated(module, hidden_states, gate):
    return normalize(module, hidden_states, gate)
