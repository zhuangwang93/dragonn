from __future__ import division
from __future__ import print_function

import time
import torch
import sys
from cuda.functions import topk_select


size = int(float(sys.argv[1])*1024*1024)
sparsity = float(sys.argv[2])
nodes = 8
k = int(size * sparsity)

device = torch.device("cuda")

kwargs = {'dtype': torch.float32,
          'device': device,
          'requires_grad': False}


x = torch.rand(size, **kwargs)
indices = torch.zeros(k, dtype=torch.int32, device=device)

top_k_samples = int(k*1.5)
threshold = torch.min(torch.topk(x, top_k_samples, 0, largest=True, sorted=False)[0])
print("size:", size, "sparsity:", sparsity, "k:", k)

torch.cuda.synchronize()
start_time = time.time()
for _ in range(nodes):
    mask = torch.ge(x, threshold)
    select_indices = torch.nonzero(mask, as_tuple=False).view(-1)
    #indices = mask.nonzero().view(-1)
    select_indices = select_indices[:k]
torch.cuda.synchronize()
mask_time = (time.time() - start_time)/nodes
print("k:", select_indices.numel(), "mask gpu time:", mask_time)


top_k_samples = int(k*2)
threshold = torch.min(torch.topk(x, top_k_samples, 0, largest=True, sorted=False)[0])

torch.cuda.synchronize()
start_time = time.time()
for _ in range(nodes):
    topk_select(x, indices, threshold)
torch.cuda.synchronize()
gpu_topk_time = (time.time() - start_time)/nodes
print("k:", indices.numel(),  "GPU topk time:", gpu_topk_time, "ratio:", gpu_topk_time/mask_time)
print()
