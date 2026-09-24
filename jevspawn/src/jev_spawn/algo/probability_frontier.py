import torch


def expand_frontier(logits, legal, parent_log_probabilities, branch_budget):
    """Expand finite actions and retain the highest-probability paths on device.

    Inputs have shapes [task, parent, action] and [task, parent]. Equal scores
    preserve parent/action order. Nonexistent padded actions remain invalid.
    This selects branches; it does not fabricate their observations or execute
    their model/environment states.
    """
    assert logits.is_cuda and legal.is_cuda and parent_log_probabilities.is_cuda
    assert logits.shape == legal.shape
    assert logits.shape[:-1] == parent_log_probabilities.shape
    scores = logits.float().masked_fill(~legal, -torch.inf)
    conditional = torch.where(legal, scores - scores.logsumexp(dim=-1, keepdim=True), -torch.inf)
    joint = conditional + parent_log_probabilities.unsqueeze(-1)
    flat = joint.flatten(start_dim=1)
    assert 0 < branch_budget <= flat.shape[-1]
    indices = flat.argsort(dim=-1, descending=True, stable=True)[:, :branch_budget]
    selected = flat.gather(-1, indices)
    return {'parents': indices // logits.shape[-1], 'actions': indices % logits.shape[-1],
            'log_probabilities': selected, 'valid': selected.isfinite()}
