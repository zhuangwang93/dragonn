import torch

from mergeComp_dl.torch import Compressor


class PoolQSGDCompressor(Compressor):
    def __init__(self, quantum_num=64):
        super().__init__()
        self.name = "PoolQSGD"
        self.quantization = True
        self.sparsification = False
        self.quantum_num = quantum_num


    def compress(self, tensor, name):
        shape = tensor.size()
        tensor = tensor.flatten()
        norm = tensor.norm().reshape((1,))

        abs_gradient = tensor.abs()

        level_float = self.quantum_num / tensor.norm() * abs_gradient
        previous_level = level_float.floor()
        prob = torch.empty_like(tensor).uniform_()
        is_next_level = (prob < (level_float - previous_level)).type(torch.float32)
        new_level = (previous_level + is_next_level)
        sign = tensor.sign()
        tensor_compressed = (new_level * sign).type(torch.int16)
        tensor_compressed = tensor_compressed.type(torch.int8 if self.quantum_num < 128 else torch.half)
        # tensor_compressed = tensor_compressed, norm
        tensor_compressed = tensor_compressed[:tensor.numel() // 2], norm
        ctx = (name, shape)
        return tensor_compressed, ctx


    def decompress(self, tensor_compressed, ctx):
        name, shape = ctx
        tensor, norm = tensor_compressed
        tensor = torch.cat((tensor, tensor), dim=0)
        # tensor = torch.cat((tensor, tensor, tensor, tensor, tensor, tensor, tensor, tensor), dim=0)
        norm = norm[0]
        decode_output = tensor.type(torch.float32)
        # return decode_output.view(shape)
        tensor_decompressed = norm / self.quantum_num * decode_output
        return tensor_decompressed.view(shape)