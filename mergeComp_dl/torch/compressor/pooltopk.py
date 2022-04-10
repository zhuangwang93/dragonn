import torch
from math import ceil
from mergeComp_dl.torch import Compressor


class PoolTopKCompressor(Compressor):
    def __init__(self, compress_ratio):
        super().__init__()
        self.name = "PoolTopK"
        self.quantization = False
        self.compress_ratio = compress_ratio


    def sparsify(self, tensor):
        k = ceil(tensor.numel() * self.compress_ratio)
        _, indices = torch.topk(tensor.abs(), k)
        values = tensor[indices]
        return values, indices.type(torch.int32)


    def desparsify(self, tensors, numel):
        values, indices = tensors
        tensor_decompressed = torch.zeros(numel, dtype=values.dtype, device=values.device)
        tensor_decompressed.scatter_(0, indices.type(torch.int64), values)
        return tensor_decompressed


    def compress(self, tensor, name):
        shape = tensor.size()
        tensor = tensor.flatten()
        values, indices = self.sparsify(tensor)
        return (values, indices), (name, shape, tensor.numel(), values.numel())


    def decompress(self, tensors, ctx):
        """Decompress by filling empty slots with zeros and reshape back using the original shape"""
        name, shape, numel, num_selects = ctx
        tensor_decompressed = self.desparsify(tensors, numel)
        return tensor_decompressed.view(shape)