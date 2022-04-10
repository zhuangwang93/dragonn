import torch
import horovod.torch as hvd

class NoneMemory():
    def __init__(self):
        self.world_size = hvd.size()

    def compensate(self, tensor, name):
        return tensor

    def update(self, tensor, name, compressor, tensor_compressed, ctx):
        """Update the residuals."""
        pass

    def aggregate(self, tensors):
        return sum(tensors)