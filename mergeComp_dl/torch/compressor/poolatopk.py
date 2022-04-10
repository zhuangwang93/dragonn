import torch
import random
import math
import os, time
import horovod.torch as hvd
import numpy as np
from math import ceil
from mergeComp_dl.torch import Compressor
from extensions.cuda.functions import topk_select


class PoolATopKCompressor(Compressor):
    def __init__(self, compress_ratio, sample_ratio=0.01, strided_sample=False,
                 compress_upper_bound=1.5, compress_lower_bound=1,
                 max_adaptation_iters=5, benchmark=False):
        super().__init__(tensors_size_are_same=False)
        self.name = "PoolATopk"
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
        self.zeros = {}
        self.benchmark = benchmark
        if self.benchmark:
            self.log_filename = os.environ['COMPRESS_LOG_FILENAME']

        # self.seeds=np.random.permutation(1000)
        self.seed_cnt=0


    def _sparsify(self, tensor, name):
        sample_ratio, compress_ratio = self.attributes
        numel = tensor.numel()
        shape = tensor.size()
        tensor = tensor.flatten()

        if numel <= 1024*128:
            compress_ratio = 0.01

        # init the tensor for decompression
        self.zeros[name] = torch.zeros(numel, dtype=tensor.dtype, device=tensor.device)
        num_samples = int(numel * sample_ratio)
        num_selects = int(numel * compress_ratio)
        indices = torch.zeros(int(num_selects) * 4, dtype=torch.int32, device=tensor.device)
    
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

        topk_select(tensor, indices, thr, seed=self.seed_cnt)
        self.seed_cnt += 1
        if self.seed_cnt >= 100000:
            self.seed_cnt=0
        
        indices = indices[indices.nonzero(as_tuple=True)]
        values = tensor[indices.type(torch.int64)]

        if hvd.rank() == 0 and self.benchmark:
            torch.cuda.synchronize()
            with open(self.log_filename, "a") as f:
                f.write("[ATopk compress] numel: {}, ratio: {}, encoding: {:.4f} ms\n".format(numel, compress_ratio, (time.time() - start_time)*1000))
        return values, indices, numel, num_selects, shape


    def compress(self, tensor, name):
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
        self.zeros[name].scatter_(0, indices.type(torch.int64), values)

        if hvd.rank() == 0 and self.benchmark:
            _, compress_ratio = self.attributes
            torch.cuda.synchronize()
            with open(self.log_filename, "a") as f:
                f.write("[compress] numel: {}, ratio: {}, encoding: {:.4f} ms\n".format(numel, compress_ratio, (time.time() - start_time)*1000))
        return self.zeros[name].view(shape)