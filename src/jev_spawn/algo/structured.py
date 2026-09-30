import torch


def padded(sequences, pad_id, device, side):
    width = max(map(len, sequences))
    ids, masks = [], []
    for sequence in sequences:
        padding = width - len(sequence)
        if side == 'left':
            ids.append([pad_id] * padding + sequence)
            masks.append([0] * padding + [1] * len(sequence))
        else:
            ids.append(sequence + [pad_id] * padding)
            masks.append([1] * len(sequence) + [0] * padding)
    return (torch.tensor(ids, device=device), torch.tensor(masks, device=device))


def common_prefix(sequences):
    length = 0
    for tokens in zip(*sequences):
        if len(set(tokens)) != 1:
            break
        length += 1
    return min(length, min(map(len, sequences)) - 1)
