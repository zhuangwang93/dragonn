import torch
from horovod.torch import allreduce_


class DgcMemory():
    def __init__(self, momentum=0.9, gradient_clipping=False, momentum_masking=False):
        self.gradient_clipping = gradient_clipping
        self.momentum = momentum
        self.momentum_masking = momentum_masking
        self.gradients = {}
        self.residuals = {}


    def compensate(self, tensor, name):
        """Update the tensor with the residuals."""
        # https://github.com/synxlin/deep-gradient-compression/blob/master/dgc/memory.py
        if self.gradient_clipping:
            tensor_squ_sum = torch.sum(tensor * tensor)
            clipping_val = torch.sqrt(allreduce_(tensor_squ_sum, average=True, name=name))
            tensor = tensor.clamp(-clipping_val, clipping_val)

        if name in self.residuals:
            self.residuals[name] = self.momentum * self.residuals[name] + tensor
        else:
            self.residuals[name] = tensor

        if not self.momentum_masking:
            return self.residuals[name]

        if name in self.gradients:
            self.gradients[name] = self.gradients[name] + self.residuals[name]
            tensor = self.gradients[name]
        else:
            self.gradients[name] = tensor
        
        return tensor


    def update(self, tensor, name, compressor, tensor_compressed, ctx):
        """Update the residuals."""
        values, indices = tensor_compressed
        name, shape, numel, num_selects = ctx
        self.residuals[name].view(-1).index_fill_(0, indices.type(torch.int64), 0)
        if self.momentum_masking:
            self.gradients[name].view(-1).index_fill_(0, indices.type(torch.int64), 0)