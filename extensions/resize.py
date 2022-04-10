import torch
import torch.nn.functional as F
import time

size = 32*1024*1024 - 1
a = torch.randn(size, dtype=torch.float32).cuda()

mask = a > 0.1
p2d = (0, 1)

torch.cuda.synchronize()
start_time = time.time()
b = F.pad(mask, p2d, 'constant', 0)
torch.cuda.synchronize()
print("convert time:", time.time()-start_time)
