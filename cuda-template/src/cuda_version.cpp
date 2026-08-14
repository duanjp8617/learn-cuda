#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>

void print_version(const char* label, int version) {
  std::cout << label << ": " << version / 1000 << '.' << (version % 1000) / 10 << " (" << version << ")\n";
}

bool check_cuda(cudaError_t error, const char* operation) {
  if (error == cudaSuccess) {
    return true;
  }

  std::cerr << operation << ": " << cudaGetErrorString(error) << '\n';
  return false;
}


int check_cuda_version() {
  int runtime_version = 0;
  int driver_version = 0;

  if (!check_cuda(cudaRuntimeGetVersion(&runtime_version), "cudaRuntimeGetVersion") ||
      !check_cuda(cudaDriverGetVersion(&driver_version), "cudaDriverGetVersion")) {
    return EXIT_FAILURE;
  }

  print_version("CUDA runtime", runtime_version);
  print_version("CUDA driver", driver_version);
  return 0;
}
