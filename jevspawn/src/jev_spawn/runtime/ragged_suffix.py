import torch


class RaggedSuffix:
    def __init__(self, lengths, device):
        self.batch_size, self.width = len(lengths), max(lengths)
        self.lengths = torch.tensor(lengths, device=device, dtype=torch.long)
        self.cu_seqlens_cpu = torch.tensor([0, *lengths], dtype=torch.int32).cumsum(0, dtype=torch.int32)
        self.cu_seqlens = self.cu_seqlens_cpu.to(device)
        self.indices = torch.tensor([row * self.width + offset
                                     for row, length in enumerate(lengths) for offset in range(length)],
                                    device=device, dtype=torch.long)

    def pack(self, tensor):
        return tensor.flatten(0, 1).index_select(0, self.indices).unsqueeze(0)

    def unpack(self, tensor):
        shape = (self.batch_size * self.width, *tensor.shape[2:])
        output = tensor.new_zeros(shape).index_copy_(0, self.indices, tensor.squeeze(0))
        return output.view(self.batch_size, self.width, *tensor.shape[2:])
