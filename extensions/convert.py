from __future__ import division
from __future__ import print_function

import time
import torch
import math
from cuda.functions import bool_to_uint8, uint8_to_bool, bits_to_int64, int64_to_bits

size = 16*1024*1024

device = torch.device("cuda")

kwargs = {'device': device,
          'requires_grad': False}

"""
x = torch.randn(size, **kwargs)
mask = x > 0.2
mask_test = mask.clone()

uint8_size = size // 8
uint8_tensor = torch.zeros(uint8_size, dtype=torch.uint8, **kwargs)

for _ in range(8):
    torch.cuda.synchronize()
    start_time = time.time()
    bool_to_uint8(mask, uint8_tensor)
    uint8_to_bool(uint8_tensor, mask)
    torch.cuda.synchronize()
    print("convert time:", time.time()-start_time)
    print(torch.eq(mask, mask_test).all())
"""
bits_size = 3

uint8_tensor = torch.randint(low=0, high=2**bits_size, size=(size,), dtype=torch.uint8, **kwargs)
uint8_test = uint8_tensor.clone()

int64_size = math.ceil(size / (64 / bits_size))
int64_tensor = torch.zeros(int64_size, dtype=torch.long, **kwargs)

bits_to_int64(uint8_tensor, bits_size, int64_tensor)
int64_to_bits(int64_tensor, bits_size, uint8_tensor)

print(torch.eq(uint8_tensor, uint8_test).all())
