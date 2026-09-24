import torch


def rank_extensions(distributions, prefix_log_probabilities, width):
    """Keep the highest-scoring extensions of the current conditional field beam."""
    probabilities = torch.stack(distributions)
    scores = prefix_log_probabilities[:, None] + probabilities.log()
    selected = scores.flatten().argsort(descending=True, stable=True)[:width]
    parents = selected // probabilities.shape[-1]
    values = selected % probabilities.shape[-1]
    return parents.tolist(), values.tolist(), scores.flatten()[selected]
