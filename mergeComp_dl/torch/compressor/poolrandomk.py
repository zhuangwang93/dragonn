import torch
from math import ceil
from mergeComp_dl.torch import Compressor


class PoolRandomKCompressor(Compressor):
    def __init__(self, compress_ratio):
        super().__init__()
        self.name = "RandomK"
        self.quantization = False
        self.compress_ratio = compress_ratio


    def sparsify(self, tensor):
        numel = tensor.numel()
        k = ceil(numel * self.compress_ratio)
        indices = torch.randint(0, numel, (k,), device=tensor.device, dtype=torch.int64)
        values = tensor[indices]

        return values, indices.type(torch.int32)


    def compress(self, tensor, name):
        shape = tensor.size()
        tensor = tensor.flatten()

        tensors = self.sparsify(tensor)
        ctx = name, tensor.numel(), shape
        return tensors, ctx


    def decompress(self, tensors, ctx):
        name, numel, shape= ctx
        values, indices = tensors
        tensor_decompressed = torch.zeros(numel, dtype=values.dtype, device=values.device)
        tensor_decompressed.scatter_(0, indices.type(torch.int64), values)
        return tensor_decompressed.view(shape)