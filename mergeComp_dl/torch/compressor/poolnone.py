from mergeComp_dl.torch import Compressor


class PoolNoneCompressor(Compressor):
    """Default no-op compression."""
    def __init__(self):
        super().__init__()
        self.name = "PoolNone"
        self.quantization = False

    def compress(self, tensor, name):
        ctx = (name, tensor.numel(), tensor.size())
        return tensor, ctx

    def decompress(self, tensors, ctx):
        return tensors
