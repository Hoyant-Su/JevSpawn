import torch
import torch.distributed as dist


def distributed_alignment(model, device, args):
    head = model.get_output_embeddings()
    output = head.weight.detach().to(device=device, dtype=torch.float32)
    inputs = model.get_input_embeddings().weight.detach().to(device=device, dtype=torch.float32)
    start = dist.get_rank(head.group) * output.shape[0]
    matching_inputs = inputs.narrow(0, start, output.shape[0])
    gram = output.T @ output
    rhs = output.T @ matching_inputs
    dist.all_reduce(gram, op=dist.ReduceOp.SUM, group=head.group)
    dist.all_reduce(rhs, op=dist.ReduceOp.SUM, group=head.group)
    gram = gram + args.alignment_settings['regularizer'] * torch.eye(
        gram.shape[0], device=gram.device, dtype=gram.dtype)
    matrix = torch.linalg.solve(gram, rhs)
    target_norm = inputs.norm(dim=1).mean().detach()
    if not args.latent_space_realign:
        matrix = torch.eye(matrix.shape[0], device=matrix.device, dtype=matrix.dtype)
    return matrix, target_norm
