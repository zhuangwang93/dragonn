import torch
from mergeComp_dl.torch import Communicator
from horovod.torch import allgather, allgather_async, synchronize, allreduce, allreduce_async_
import time
import horovod.torch as hvd
from threading import Thread
from queue import Empty, Queue
import os
from horovod.torch import Average

class PoolAllgather(Communicator):
    def __init__(self, compressor, memory, batching=False, benchmark=False):
        super().__init__(compressor, memory)
        self.name = "PoolAllGather"
        self.handles = {}
        self.signals = {}
        self.device = torch.cuda.current_device()
        # self.queue = Queue()
        # self.allgather_thread = Thread(target=self.allgather_consumer)
        # self.allgather_thread.start()
        self.world_size = hvd.size()
        self.rank = hvd.rank()
        self.batching = batching
        self.benchmark = benchmark
        if self.benchmark:
            self.log_filename = os.environ['COMPRESS_LOG_FILENAME']
            assert(self.log_filename != ""), "the log filename has not been set yet."
            print("Benchmark the compress overhead and communication overhead.\n The log is written into {}".format(self.log_filename))
        

    def allgather_consumer(self):
        torch.cuda.set_device(self.device)
        print("allgather consumer is running...")
        while int(os.environ['RUN_HOROVOD']):
            try:
                ctx = self.queue.get(timeout=5)
                self.allgather_synchronize(ctx)
            except Empty:
                pass


    def allgather_synchronize(self, ctx):
        tensors_compressed, ctx = ctx
        name = ctx[0]
        if self.benchmark:
            torch.cuda.synchronize()
            start_time = time.time()

        # when allreduce_aysnc() is called in allgather_synchronize() and synchronize() in wait_receive(), the accuracy is as expected
        # however, if both allreduce_aysnc() and synchronize() are called in allgather_synchronize(), it harms the accuracy.
        # still have no clue why it comes.
        # We also observe similar results for allgather_async()

        # full gradients with allgather works
        # handle = allgather_async(tensors_compressed, name=name+"allgather")
        # full gradients with allreduce works
        # handle = allreduce_async_(tensors_compressed, average=Average, name=name)
        # self.handles[name].append(handle)

        for i, tensor_compressed in enumerate(tensors_compressed):
            # print("[allgather_synchronize]", name, tensor_compressed.numel(), hvd.rank(), flush=True)
            self.handles[name].append(allgather_async(tensor_compressed, name + str(i)))

        if hvd.rank() == 0 and self.benchmark:
            torch.cuda.synchronize()
            with open(self.log_filename, "a") as f:
                f.write("[Comm] GPUs: {}, allgather: {:.4f} ms\n".format(self.world_size, (time.time() - start_time)*1000))
        self.signals[name] = 1
        return self.handles[name]


    def async_send(self, tensors_compressed, ctx):
        # assert(len(tensors_compressed) == 2)
        name = ctx[0]
        self.handles[name] = []
        self.signals[name] = 0

        # self.queue.put((tensors_compressed, ctx))
        handles = self.allgather_synchronize((tensors_compressed, ctx))
        return handles, ctx


    def wait_receive(self, handles, ctx):
        name = ctx[0]
        while self.signals[name] == 0:
            time.sleep(0.5/1000)
        handles = self.handles[name]
        values, indices = synchronize(handles[0]), synchronize(handles[1])

        if self.batching:
            output = self.compressor.decompress((values, indices), ctx)
            return output #/ self.world_size
        else:
            values, indices = values.chunk(self.world_size), indices.chunk(self.world_size)
            tensors_decompressed = []
            for value, index in zip(values, indices):
                tensors_decompressed.append(self.compressor.decompress((value, index), ctx))

            return sum(tensors_decompressed) #/ self.world_size