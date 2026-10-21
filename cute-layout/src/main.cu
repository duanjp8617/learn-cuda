// #include <cute/tensor.hpp>

// #include <iostream>
// #include <cute/tensor.hpp>

// #include "numeric.hpp"
// #include "container.hpp"
// #include "util.hpp"
// #include "int_tuple.hpp"
// #include "stride.hpp"
// #include "arithmetic_tuple.hpp"
// #include "t1_layout.hpp"
// #include "t2_layout_algebra.hpp"
#include <cuda_runtime.h>
#include <iostream>

#include "t3_tensor.hpp"

int main() {
  float *data = nullptr;
  cudaError_t err =
      cudaMalloc(reinterpret_cast<void **>(&data), 1024 * sizeof(float));

  if(err != cudaSuccess) {
    std::cout<<"can't allocate memory\n";
    return -1;
  }

  run_testor<<<1, 1>>>(data, 16);

  err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cerr<<"Kernel launch failure: "
            << cudaGetErrorString(err)
            << "\n";
    return 1;
  }

  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    std::cerr<<"Kernel execution failure: "
            << cudaGetErrorString(err)
            << "\n";
    return 1;
  }

  return 0;
}

int _main() {
  test_layout_slice();
}