from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import torch
import time
import random
import horovod.torch as hvd
from statistics import mean

has_gpu = True
hvd.init()

class CompressorTest:
    """
    Tests for mergeComp compressor
    """

    def __init__(self, comp, compress_ratio=0.001, ranks=1):
        self.set_current_context()
        self.compress_ratio = compress_ratio
        self.compressor = self.set_compressor(comp)
        self.ranks = ranks
        self.residual = self.init_tensor()
        self.tensor = self.init_tensor()


    def set_current_context(self):
        if has_gpu:
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

        self.kwargs = {'dtype': torch.float32,
                'device': device,
                'requires_grad': False}


    def set_compressor(self, comp):
        sys.path.append("../")
        if comp == 'dgc':
            from mergeComp_dl.torch.compressor.pooldgc import PoolDgcCompressor
            return PoolDgcCompressor(compress_ratio=self.compress_ratio)
        elif comp == 'atopk':
            from mergeComp_dl.torch.compressor.poolatopk import PoolATopKCompressor
            return PoolATopKCompressor(compress_ratio=self.compress_ratio)
        elif comp == 'topk':
            from mergeComp_dl.torch.compressor.pooltopk import PoolTopKCompressor
            return PoolTopKCompressor(compress_ratio=self.compress_ratio)
        else:
            raise NotImplementedError(comp)


    def init_tensor(self, size=2**26):
        return torch.rand(size, **self.kwargs)


    def compress(self, tensor):
        name = str(tensor.numel())
        tensor_compressed, ctx = self.compressor.compress(tensor, name)
        return tensor_compressed, ctx 


    def decompress(self, tensor, ctx):
        return self.compressor.decompress(tensor, ctx)

    
    def test_compressor(self, size, start=0):
        return self.compress(self.tensor[start:start+size])

        
    def test_decompressor(self, tensor, ctx):
        return self.decompress(tensor, ctx)


def test_compressor(comp, rate):
    compressor = CompressorTest(comp=comp, compress_ratio=rate)
    sizes = []
    encode_times = []
    for p in range(17, 18):
        size = 2 ** p
        sizes.append(size)
        torch.cuda.synchronize()
        compress_time, decompress_time = 0, 0
        skip = 5
        size_encode_times = []

        for i in range(runs):
            start = random.randint(0, 2**14)
            torch.cuda.synchronize()
            start_timestamp = time.time()
            tensor_compressed, ctx = compressor.test_compressor(size, start)
            torch.cuda.synchronize()
            compress_timestamp = time.time()
            compress_time = compress_timestamp - start_timestamp
            size_encode_times.append(compress_time)
            time.sleep(0.01)

            # compressor.test_decompressor(tensor_compressed, ctx)
            # torch.cuda.synchronize()
            # decompress_timestamp = time.time()
            # if i >= skip:
            #     decompress_time += decompress_timestamp - compress_timestamp
        encode_times.append(round(mean(size_encode_times[skip:])*1000, 3))

    print(comp, sizes, rate, encode_times, flush=True)

# the encoding+decoding overhead of FP16 is around 0.035ms
runs = 100

if __name__ == "__main__":
    for comp in ["dgc", "atopk"]:
        for rate in [0.001]:
            print("compressor:", comp, rate)
            test_compressor(comp, rate)
