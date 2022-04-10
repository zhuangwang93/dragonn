from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='functions_cuda',
    ext_modules=[
        CUDAExtension('functions_cuda', [
            'functions_cuda.cpp',
            'functions_cuda_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
