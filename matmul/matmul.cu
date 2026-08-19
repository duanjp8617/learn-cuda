#include "matmul.hpp"

#define CEIL_DIV(N, div) ((N + div - 1) / div)

// Compute (M, K) @ (N, K)
template <int M, int N, int K, int BM, int BN, int BK>
__global__ void matmul_1d_tiling(const float *A, const float *B, float *C) {
  // block size must be BM*BN

  // starting address of the A tile, a (BM, K) submatrix
  const float *A_tile = A + blockIdx.y * BM * K;
  const int M_remaining = M - blockIdx.y * BM;
  // starting address of the B tile, a (K, BN) submatrix
  const float *B_tile = B + blockIdx.x * BN;
  const int N_remaining = N - blockIdx.x * BN;
  // Starting address of the C tile for writing back
  float *C_tile = C + blockIdx.y * BM * N + blockIdx.x * BN;

  // shared memory for storing a portion of the A/B tile
  __shared__ float A_tile_cache[BM][BK];
  __shared__ float B_tile_cache[BK][BN];

  int row = threadIdx.x / BN;
  int col = threadIdx.x % BN;
  float accum = 0.0;

  // load A/B tile into the shared memory
  for (int k = 0; k < CEIL_DIV(K, BK); k++) {
    int K_remaining = K - k * BK;

    for (int tid = threadIdx.x; tid < BM * BK; tid += BM * BN) {
      int row = tid / BK;
      int col = tid % BK;
      A_tile_cache[row][col] =
          row < M_remaining && col < K_remaining
              ? A_tile[row * K + k * BK + col]
              : 0.0F;
    }

    for (int tid = threadIdx.x; tid < BK * BN; tid += BM * BN) {
      int row = tid / BN;
      int col = tid % BN;
      B_tile_cache[row][col] =
          row < K_remaining && col < N_remaining
              ? B_tile[(k * BK + row) * N + col]
              : 0.0F;
    }

    __syncthreads();

    for (int k = 0; k < BK; k++) {
      accum += A_tile_cache[row][k] * B_tile_cache[k][col];
    }

    __syncthreads();
  }

  if (row < M_remaining && col < N_remaining) {
    C_tile[row * N + col] = accum;
  }
}

template __global__ void matmul_1d_tiling<37, 29, 53, 16, 16, 16>(const float *, const float *,
                                                                     float *);
template __global__ void matmul_1d_tiling<1024, 1024, 1024, 16, 16, 16>(
    const float *, const float *, float *);
