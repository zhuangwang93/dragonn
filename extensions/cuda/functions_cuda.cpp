#include <torch/extension.h>

#include <vector>
#include <iostream>

// CUDA forward declarations
void topk_select_cuda (void* input, int input_size, void* indices, int indices_size, float thres, int seed);
// C++ interface

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


void topk_select (
        torch::Tensor input,
        torch::Tensor indices,
        float thres,
        int seed) {
    CHECK_INPUT(input);
    CHECK_INPUT(indices);

    int input_size = input.numel();
    int indices_size = indices.numel();
    topk_select_cuda(input.data_ptr(), input_size, indices.data_ptr(), indices_size, thres, seed);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("topk_select", &topk_select, "Top K select (CUDA)");
}
