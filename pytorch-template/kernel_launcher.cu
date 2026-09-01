#include <torch/extension.h>
#include "kernel_traits.h"
#include "vector_add.h"

template<typename Traits>
void kernel_template(float* a, float* b, float* c, int N) {
    int block_dim = (N + Traits::block_size - 1) / Traits::block_size;

    vector_add<Traits><<<block_dim, Traits::block_size>>>(a, b, c, N);
}

inline void specialized_kernel(float* a, float* b, float* c, int N) {
    using launch_traits = VectorAddTrait<256>;
    kernel_template<launch_traits>(a, b, c, N);
}

torch::Tensor vector_add(torch::Tensor a, torch::Tensor b) {
    // Check cuda tensors
    TORCH_CHECK(a.device().is_cuda() && b.device().is_cuda(), "non-cuda device tensor");
    TORCH_CHECK(a.numel() == b.numel(), "invalid tensor size");

    // For a basic add kernel, we allocate the output tensor locally
    torch::Tensor output = torch::empty(a.numel(), a.options());

    specialized_kernel(a.data_ptr<float>(),b.data_ptr<float>(), output.data_ptr<float>(), a.numel());
}

// This is required to build the required python module
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("add", &add, "Add two vectors, return a new vector"); }