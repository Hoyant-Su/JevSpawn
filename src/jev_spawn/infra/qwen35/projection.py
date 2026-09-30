import torch
import torch.nn.functional as F


@torch.no_grad()
def pack_input_projections(module, names):
    projections = [getattr(module, name) for name in names]
    sizes = tuple(projection.out_features for projection in projections)
    weight = torch.cat([projection.weight.detach() for projection in projections])
    for projection, view in zip(projections, weight.split(sizes), strict=True):
        projection.weight.data = view
    if all(projection.bias is None for projection in projections):
        bias = None
    elif all(projection.bias is not None for projection in projections):
        bias = torch.cat([projection.bias.detach() for projection in projections])
        for projection, view in zip(projections, bias.split(sizes), strict=True):
            projection.bias.data = view
    else:
        raise ValueError('Packed input projections require consistently present or absent biases.')
    module.register_buffer('_input_projection_weight', weight, persistent=False)
    module.register_buffer('_input_projection_bias', bias, persistent=False)
    module._input_projection_sizes = sizes


def input_projections(module, hidden_states):
    return F.linear(hidden_states, module._input_projection_weight,
                    module._input_projection_bias).split(module._input_projection_sizes, dim=-1)
