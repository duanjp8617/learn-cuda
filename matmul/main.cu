#include "matmul.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

void check_cuda(cudaError_t status, const char *expression, const char *file, int line) {
  if (status != cudaSuccess) {
    std::fprintf(stderr, "CUDA error for %s at %s:%d: %s\n", expression, file, line,
                 cudaGetErrorString(status));
    std::exit(EXIT_FAILURE);
  }
}

#define CHECK_CUDA(expression) check_cuda((expression), #expression, __FILE__, __LINE__)

template <int M, int N, int K, int BM, int BN, int BK>
void launch_matmul(const float *device_a, const float *device_b, float *device_c) {
  const dim3 block(BM * BN);
  const dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
  matmul_1d_tiling<M, N, K, BM, BN, BK><<<grid, block>>>(device_a, device_b, device_c);
}

template <int M, int N, int K>
void initialize_inputs(std::vector<float> &a, std::vector<float> &b) {
  for (int row = 0; row < M; ++row) {
    for (int col = 0; col < K; ++col) {
      a[static_cast<size_t>(row) * K + col] =
          static_cast<float>(((row * 13 + col * 7) % 17) - 8) / 17.0F;
    }
  }
  for (int row = 0; row < K; ++row) {
    for (int col = 0; col < N; ++col) {
      b[static_cast<size_t>(row) * N + col] =
          static_cast<float>(((row * 5 + col * 11) % 19) - 9) / 19.0F;
    }
  }
}

template <int M, int N, int K>
bool verify_result(const std::vector<float> &a, const std::vector<float> &b,
                   const std::vector<float> &c) {
  float max_error = 0.0F;
  for (int row = 0; row < M; ++row) {
    for (int col = 0; col < N; ++col) {
      float expected = 0.0F;
      for (int inner = 0; inner < K; ++inner) {
        expected += a[static_cast<size_t>(row) * K + inner] *
                    b[static_cast<size_t>(inner) * N + col];
      }
      max_error = std::max(max_error,
                           std::abs(c[static_cast<size_t>(row) * N + col] - expected));
    }
  }
  std::printf("Correctness: %s (max absolute error %.8f)\n",
              max_error <= 1.0e-5F ? "PASS" : "FAIL", max_error);
  return max_error <= 1.0e-5F;
}

template <int M, int N, int K, int BM, int BN, int BK>
bool run_correctness_test() {
  std::vector<float> host_a(static_cast<size_t>(M) * K);
  std::vector<float> host_b(static_cast<size_t>(K) * N);
  std::vector<float> host_c(static_cast<size_t>(M) * N);
  initialize_inputs<M, N, K>(host_a, host_b);

  float *device_a = nullptr;
  float *device_b = nullptr;
  float *device_c = nullptr;
  CHECK_CUDA(cudaMalloc(&device_a, host_a.size() * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&device_b, host_b.size() * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&device_c, host_c.size() * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(device_a, host_a.data(), host_a.size() * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(device_b, host_b.data(), host_b.size() * sizeof(float), cudaMemcpyHostToDevice));

  launch_matmul<M, N, K, BM, BN, BK>(device_a, device_b, device_c);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaMemcpy(host_c.data(), device_c, host_c.size() * sizeof(float), cudaMemcpyDeviceToHost));

  CHECK_CUDA(cudaFree(device_a));
  CHECK_CUDA(cudaFree(device_b));
  CHECK_CUDA(cudaFree(device_c));
  return verify_result<M, N, K>(host_a, host_b, host_c);
}

template <int M, int N, int K, int BM, int BN, int BK>
void run_benchmark() {
  constexpr int warmup_iterations = 5;
  constexpr int timed_iterations = 50;
  std::vector<float> host_a(static_cast<size_t>(M) * K);
  std::vector<float> host_b(static_cast<size_t>(K) * N);
  initialize_inputs<M, N, K>(host_a, host_b);

  float *device_a = nullptr;
  float *device_b = nullptr;
  float *device_c = nullptr;
  CHECK_CUDA(cudaMalloc(&device_a, host_a.size() * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&device_b, host_b.size() * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&device_c, static_cast<size_t>(M) * N * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(device_a, host_a.data(), host_a.size() * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(device_b, host_b.data(), host_b.size() * sizeof(float), cudaMemcpyHostToDevice));

  for (int iteration = 0; iteration < warmup_iterations; ++iteration) {
    launch_matmul<M, N, K, BM, BN, BK>(device_a, device_b, device_c);
  }
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));
  CHECK_CUDA(cudaEventRecord(start));
  for (int iteration = 0; iteration < timed_iterations; ++iteration) {
    launch_matmul<M, N, K, BM, BN, BK>(device_a, device_b, device_c);
  }
  CHECK_CUDA(cudaEventRecord(stop));
  CHECK_CUDA(cudaEventSynchronize(stop));

  float elapsed_ms = 0.0F;
  CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));
  const double average_ms = elapsed_ms / timed_iterations;
  const double gflops = 2.0 * M * K * static_cast<double>(N) / average_ms * 1.0e-6;
  std::printf("matmul_1d_tiling\n");
  std::printf("  C(%d,%d) = A(%d,%d) @ B(%d,%d), block %dx%d, BK %d\n", M, N, M, K, K, N,
              BM, BN, BK);
  std::printf("  Average time: %.3f ms (%d iterations)\n", average_ms, timed_iterations);
  std::printf("  Throughput: %.2f GFLOP/s\n", gflops);

  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));
  CHECK_CUDA(cudaFree(device_a));
  CHECK_CUDA(cudaFree(device_b));
  CHECK_CUDA(cudaFree(device_c));
}

}  // namespace

int main() {
  constexpr int BM = 16;
  constexpr int BN = 16;
  constexpr int BK = 16;

  if (!run_correctness_test<37, 29, 53, BM, BN, BK>()) {
    return EXIT_FAILURE;
  }
  run_benchmark<1024, 1024, 1024, BM, BN, BK>();
  return EXIT_SUCCESS;
}
