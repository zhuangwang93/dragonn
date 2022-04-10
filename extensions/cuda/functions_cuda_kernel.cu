#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace {


#define FORCE_INLINE
__device__ static inline FORCE_INLINE uint64_t rotl32 ( uint32_t x, int8_t r) {
  return (x << r) | (x >> (32 - r));
}


#define	ROTL32(x,y)	rotl32(x,y)

//-----------------------------------------------------------------------------
// Block read - if your platform needs to do endian-swapping or can only
// handle aligned reads, do the conversion here

#define getblock(p, i) (p[i])
#define SEED 43

//-----------------------------------------------------------------------------
// Finalization mix - force all bits of a hash block to avalanche
__device__ static inline FORCE_INLINE uint32_t fmix32 ( uint32_t h )
{
  h ^= h >> 16;
  h *= 0x85ebca6b;
  h ^= h >> 13;
  h *= 0xc2b2ae35;
  h ^= h >> 16;

  return h;
}

// murmurhash for integer only
// https://github.com/aappleby/smhasher/blob/master/src/MurmurHash3.cpp
__device__ uint32_t MurmurHash3_x86_32 (
        const int key, 
        const uint32_t seed) {
    uint32_t h1 = seed;
    uint32_t c1 = 0xcc9e2d51;
    uint32_t c2 = 0x1b873593;
    
    uint32_t k1 = key;
    
    k1 *= c1;
    k1 = ROTL32(k1, 15);
    k1 *= c2;
    
    h1 ^= k1;
    h1 = ROTL32(h1, 13); 
    h1 = h1*5+0xe6546b64;

    //----------
    // tail

    const uint8_t * tail = (const uint8_t*)key;
    k1 = 0;

    int len = 4;
    switch(len & 3) {
        case 3: k1 ^= tail[2] << 16;
        case 2: k1 ^= tail[1] << 8;
        case 1: k1 ^= tail[0];
        k1 *= c1; k1 = ROTL32(k1, 15); k1 *= c2; h1 ^= k1;
    };

    //----------
    // finalization
    h1 ^= len;
    h1 = fmix32(h1);
    return h1;
} 


__device__ float abs(float value) {
    return value >= 0 ? value : -value;
}

__device__ uint32_t hash(const int key) {
    uint32_t a = 0xcc9e2d51;
    uint32_t b = 0x1b873593;
    uint32_t c = 0xe6546b64;

    return a*key*key + b*key + c;
}


__global__ void topk_select_cuda_kernel (float* input, int input_size, int* indices, int indices_size, float thres, int seed) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < input_size) {
        if (abs(input[tid]) >= thres) {
            int index = MurmurHash3_x86_32(tid, SEED+seed) % indices_size;
            indices[index] = tid;
        }
    }
}

} // namespace


void topk_select_cuda (
        void* input, 
        int input_size,
        void* indices, 
        int indices_size,
        float thres,
        int seed) {
    const int threads = 512;
    const int blocks = (input_size + threads - 1) / threads;

    topk_select_cuda_kernel<<<blocks, threads>>>((float*)input, input_size, (int*)indices, indices_size, thres, seed);
}