import torch
import functions_cuda as func


def topk_select(tensor, indices, thres, seed=137):
    assert(isinstance(tensor, torch.cuda.FloatTensor))
    assert(isinstance(indices, torch.cuda.IntTensor))
    func.topk_select(tensor, indices, thres, seed)