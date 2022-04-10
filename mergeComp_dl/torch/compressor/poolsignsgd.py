import torch
from math import ceil
from mergeComp_dl.torch import Compressor
from mergeComp_dl.torch.util import packbits, unpackbits


class PoolSignSGDCompressor(Compressor):
    def __init__(self):
        super().__init__()
        self.name = "PoolEFSignSGD"
        self.quantization = True
        self.sparsification = False
        self.zeros = torch.zeros([1024], dtype=torch.bool, device=torch.cuda.current_device())


    def compress(self, tensor, name, signsgd_unit_size=8):
        """Encoding and compressing the signs """
        numel = tensor.numel()
        shape = tensor.size()
        tensor = tensor.flatten()
        
        sign_encode = tensor >= 0
        mean = tensor.abs().mean().reshape((1,))

        # for alltoall operation, we ensure all partitions have the same number of gradients after compression
        # our solution is to add padding
        unit_size = signsgd_unit_size
        padding = numel % unit_size
        if padding > 0:
            padding = unit_size - padding
            sign_encode = torch.cat([sign_encode, self.zeros[:padding]], dim=0)
        packed_tensor = packbits(sign_encode, unit_size=signsgd_unit_size)
        tensor_compressed = packed_tensor, mean
        numel_per_node = numel + padding
        # note that after padding, the compressed message in the last node has some additional zero values
        # we will get rid of them after decompress all the messages
        ctx = (name, numel, numel_per_node, shape)
        return tensor_compressed, ctx


    def decompress(self, tensor_compressed, ctx):
        """Decoding the signs to float format """
        packed_tensor, mean = tensor_compressed
        mean = mean[0]
        
        _, numel, _, shape = ctx
        sign_decode = unpackbits(packed_tensor, numel)
        tensor_decoded = mean*(sign_decode.type(torch.float32) * 2 - 1)
        
        return tensor_decoded.view(shape)