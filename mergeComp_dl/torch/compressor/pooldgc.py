import torch
import random
from math import ceil
import os, time
import horovod.torch as hvd
import numpy

from mergeComp_dl.torch import Compressor


class PoolDgcCompressor(Compressor):
    def __init__(self, compress_ratio, sample_ratio=0.01, strided_sample=True,
                 compress_upper_bound=1.3, compress_lower_bound=1.0,
                 max_adaptation_iters=10, benchmark=False):
        super().__init__(tensors_size_are_same=False)
        self.name = "PoolDGC"
        self.quantization = False
        self.compress_ratio = compress_ratio
        self.base_compress_ratio = self.compress_ratio = \
            compress_ratio if compress_ratio <= 1.0 else 1.0 / compress_ratio
        self.sample_ratio = min(max(sample_ratio, 0.001), 1.0)
        self.strided_sample = strided_sample
        self.compress_upper_bound = compress_upper_bound
        self.compress_lower_bound = compress_lower_bound
        self.max_adaptation_iters = max_adaptation_iters
        self.attributes = (self.sample_ratio, self.base_compress_ratio)
        self.benchmark = benchmark
        if self.benchmark:
            self.log_filename = os.environ['COMPRESS_LOG_FILENAME']


    def _sparsify(self, tensor, name):
        sample_ratio, compress_ratio = self.attributes
        numel = tensor.numel()
        shape = tensor.size()
        tensor = tensor.flatten()

        # init the tensor for decompression
        if numel <= 1024*128:
            compress_ratio = 0.01

        if numel <= 1024*16:
            k = ceil(numel * compress_ratio)
            _, indices = torch.topk(tensor.abs(), k)
            values = tensor[indices]
            return values, indices, numel, k, shape

        num_samples = int(numel * sample_ratio)
        num_selects = int(numel * compress_ratio)

        if self.strided_sample:
            sample_stride = int(1 // sample_ratio)
            sample_start = random.randint(0, min(sample_stride, numel-1))
            samples = tensor[sample_start::sample_stride]
        else:
            samples = tensor[torch.randint(0, numel, (num_samples, ), device=tensor.device)]

        k = ceil(num_samples * compress_ratio)
        thr = torch.min(torch.topk(samples.abs(), k, 0, largest=True, sorted=False)[0])

        if self.benchmark:
            torch.cuda.synchronize()
            start_time = time.time()

        mask = tensor.abs() >= thr
        selected = mask.sum()

        for _ in range(self.max_adaptation_iters):
            if selected > self.compress_upper_bound * num_selects:
                thr = thr / 0.9
            elif selected < self.compress_lower_bound * num_selects:
                thr = thr * 0.9
            else:
                break
            mask = tensor.abs() >= thr
            selected = mask.sum()

        indices, = torch.where(mask)
        indices = indices[:num_selects]
        values = tensor[indices]

        if hvd.rank() == 0 and self.benchmark:
            torch.cuda.synchronize()
            with open(self.log_filename, "a") as f:
                f.write("[DGC compress] numel: {}, ratio: {}, encoding: {:.4f} ms\n".format(numel, compress_ratio, (time.time() - start_time)*1000))
        return values, indices, numel, num_selects, shape


    def compress(self, tensor, name):
        if self.compress_ratio >= 1.0:
            numel = tensor.numel()
            indices = torch.from_numpy(numpy.arange(numel)).cuda()
            return (tensor.flatten(), indices.type(torch.int32)), (name, tensor.size(), numel, numel)

        values, indices, numel, num_selects, shape = self._sparsify(tensor, name)
        ctx = (name, shape, numel, num_selects)

        return (values, indices.type(torch.int32)), ctx


    def decompress(self, tensor_compressed, ctx):
        if self.benchmark:
            torch.cuda.synchronize()
            start_time = time.time()

        name, shape, numel, num_selects = ctx
        # decompress
        values, indices = tensor_compressed

        # if self.compress_ratio >= 1.0:
        #     return values.view(shape)

        tensor_decompressed = torch.zeros(numel, dtype=values.dtype, device=values.device)
        tensor_decompressed.scatter_(0, indices.type(torch.int64), values)

        if hvd.rank() == 0 and self.benchmark:
            _, compress_ratio = self.attributes
            torch.cuda.synchronize()
            with open(self.log_filename, "a") as f:
                f.write("[compress] numel: {}, ratio: {}, encoding: {:.4f} ms\n".format(numel, compress_ratio, (time.time() - start_time)*1000))
        return tensor_decompressed.view(shape)