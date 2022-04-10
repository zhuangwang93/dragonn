from torch.utils.cpp_extension import load
lltm_cuda = load(
    'functions_cuda', ['functions_cuda.cpp', 'functions_cuda_kernel.cu'], verbose=True)
help(functions_cuda)
