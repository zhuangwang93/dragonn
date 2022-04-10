import os, sys
import torch
import horovod.torch as hvd


def grace_from_params(params):
    comp = params.get('compressor', 'none')
    mem = params.get('memory', 'none')
    comm = params.get('communicator', 'allgather')
    model_params = params.get('params', 'none')
    ratio = params.get('ratio', 0.01)
    benchmark = params.get('benchmark', False)
    threshold = params.get('threshold', 2**20)
    if model_params == 'none':
        sys.exit("No model parameters for grace_from_params()")
    print("[Compression Setup] GPUs: {}\n\tcompressor: {}\n\tmemory: {}\n\tcommunicator: {}\n\tsparsity ratio: {}\n\tthreshold: {}".format(
        hvd.size(),
        comp,
        mem,
        comm,
        ratio,
        threshold,
    ))

    if comp == 'dgc':
        from mergeComp_dl.torch.compressor.pooldgc import PoolDgcCompressor
        compressor = PoolDgcCompressor(compress_ratio=ratio, benchmark=benchmark)
    elif comp == 'fp16':
        from mergeComp_dl.torch.compressor.poolfp16 import PoolFP16Compressor
        compressor = PoolFP16Compressor()
    elif comp == 'none':
        from mergeComp_dl.torch.compressor.poolnone import PoolNoneCompressor
        compressor = PoolNoneCompressor()
    elif comp == 'randomk':
        from mergeComp_dl.torch.compressor.poolrandomk import PoolRandomKCompressor
        compressor = PoolRandomKCompressor(compress_ratio=ratio)
    elif comp == 'signsgd':
        from mergeComp_dl.torch.compressor.poolsignsgd import PoolSignSGDCompressor
        compressor = PoolSignSGDCompressor()
    elif comp == 'qsgd':
        from mergeComp_dl.torch.compressor.poolqsgd import PoolQSGDCompressor
        compressor = PoolQSGDCompressor()
    elif comp == 'topk':
        from mergeComp_dl.torch.compressor.pooltopk import PoolTopKCompressor
        compressor = PoolTopKCompressor(compress_ratio=ratio)
    elif comp == 'atopk':
        from mergeComp_dl.torch.compressor.poolatopk import PoolATopKCompressor
        compressor = PoolATopKCompressor(compress_ratio=ratio, benchmark=benchmark)
    else:
        raise NotImplementedError(comp)

    if mem == 'dgc':
        from mergeComp_dl.torch.memory.dgc import DgcMemory
        memory = DgcMemory()
    elif mem == 'residual':
        from mergeComp_dl.torch.memory.residual import ResidualMemory
        memory = ResidualMemory()
    elif mem == 'none':
        from mergeComp_dl.torch.memory.none import NoneMemory
        memory = NoneMemory()
    else:
        raise NotImplementedError(mem)

    if comm == 'allreduce':
        from mergeComp_dl.torch.communicator.pool_allreduce import PoolAllreduce
        return PoolAllreduce(compressor, memory)
    elif comm == 'allgather':
        batching = params.get('batching', False)
        from mergeComp_dl.torch.communicator.pool_allgather import PoolAllgather
        return PoolAllgather(compressor, memory, batching, benchmark)
    else:
        raise NotImplementedError(comm)


def add_parser_arguments(parser):
    parser.add_argument('--use-adasum', action='store_true', default=False,
                        help='use adasum algorithm to do reduction')
    parser.add_argument('--gradient-predivide-factor', type=float, default=1.0,
                        help='apply gradient predivide factor in optimizer (default: 1.0)')
    parser.add_argument('--fp16-allreduce', action='store_true', default=False,
                        help='use fp16 compression during allreduce')
    parser.add_argument('--compress', action='store_true', default=False,
                        help='use gradient compression')
    parser.add_argument('--compressor', type=str, default='dgc',
                        help='compress algorithm')
    parser.add_argument('--compress-ratio', type=float, default=0.001,
                        help='compress ratio for sparsification')
    parser.add_argument('--memory', type=str, default='residual',
                        help='compress algorithm')
    parser.add_argument('--comm', type=str, default='allgather',
                        help='communication for compression')
    parser.add_argument('--batching', action='store_true', default=False,
                        help='batching the decoding operations')
    parser.add_argument('--pcie', action='store_true', default=False,
                        help='use pcie for GPU communication')
    parser.add_argument('--benchmark', action='store_true', default=False,
                        help='measure encoding latency')
    parser.add_argument('--threshold', type=int, default=2**20,
                        help='the threshold of tensors for compression')
    parser.add_argument('--compress-iter', type=int, default=200,
                        help='the threshold of tensors for compression')
    return parser


def wrap_compress_optimizer(model, optimizer, args):
    if args.pcie:
        os.environ["NCCL_P2P_DISABLE"] = "1"

    if args.compress:
        """
        compressor: dgc, efsignsgd, fp16, none, onebit, qsgd, randomk, signsgd, signum, terngrad, threshold, topk
        memory: dgc, none, residual, 1bitadam.   Note: 1bitadam is for Adam
        comm: allreduce, allgather, ps
        """
        params = {
            'compressor': args.compressor,
            'memory': args.memory,
            'communicator': args.comm,
            'params': model.named_parameters(),
            'ratio': args.compress_ratio,
            'batching': args.batching,
            'benchmark': args.benchmark,
            'threshold': args.threshold
        }

        os.environ['RUN_HOROVOD'] = "1"
        grc = grace_from_params(params)

        from .horovod.optimizer import DistributedOptimizer
        optimizer = DistributedOptimizer(optimizer, compression=grc, threshold=args.threshold, compression_start_iter=args.compress_iter, named_parameters=model.named_parameters())

        return optimizer
    else:
        # Horovod: (optional) compression algorithm.
        compression = hvd.Compression.fp16 if args.fp16_allreduce else hvd.Compression.none
        optimizer = hvd.DistributedOptimizer(optimizer,
                                         named_parameters=model.named_parameters(),
                                         compression=compression,
                                         op=hvd.Adasum if args.use_adasum else hvd.Average,
                                         gradient_predivide_factor=args.gradient_predivide_factor)
        return optimizer


def stop_training():
    os.environ['RUN_HOROVOD'] = "0"