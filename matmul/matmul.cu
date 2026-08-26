#include "matmul.hpp"

#define CEIL_DIV(N, div) ((N + div - 1) / div)

// Compute (M, K) @ (N, K)
// 
// Implement block tiling, where block threads collectively compute the 
// result through shared memory. 
//
// 
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
      A_tile_cache[row][col] = row < M_remaining && col < K_remaining
                                   ? A_tile[row * K + k * BK + col]
                                   : 0.0F;
    }

    for (int tid = threadIdx.x; tid < BK * BN; tid += BM * BN) {
      int row = tid / BN;
      int col = tid % BN;
      B_tile_cache[row][col] = row < K_remaining && col < N_remaining
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

template __global__ void
matmul_1d_tiling<37, 29, 53, 16, 16, 16>(const float *, const float *, float *);
template __global__ void
matmul_1d_tiling<1024, 1024, 1024, 16, 16, 16>(const float *, const float *,
                                               float *);

// The 2d thread tiling, where the compute intensity of a thread is incresased by 
// computing the TM*TN results.
__global__ void matmul_2d_tiling(int M, int N, int K, const float *A,
                                 const float *B, float *C) {
  // Define basic operation sizes
  constexpr int BM = 128;
  constexpr int BN = 128;
  constexpr int BK = 32;
  constexpr int TM = 8;
  constexpr int TN = 8;

  constexpr int BLOCK_COL_SIZE = BN / TN;
  static_assert(BN % TN == 0, "invalid block size");

  // 2 dimentional blocks.
  // Each block contains BM*BN threads
  const int block_size = (BM / TM) * (BN / TN);

  // A stripe, a BM * K stripe area on A
  const float *A_stripe = A + blockIdx.y * BM * K;
  const int M_remaining = M - blockIdx.y * BM;

  // B stripe, a K * BN stripe area on B
  const float *B_stripe = B + blockIdx.x * BN;
  const int N_remaining = N - blockIdx.x * BN;

  float *C_tile = C + blockIdx.y * BM * N + blockIdx.x * BN;

  // Prepare the shared memory cache on each block.
  __shared__ float A_tile_cache[BM][BK];
  __shared__ float B_tile_cache[BK][BN];

  float TA_tile_cache[TM];
  float TB_tile_cache[TN];
  float TC_tile_cache[TM][TN] = {0.0};

  int threadrow = threadIdx.x / BLOCK_COL_SIZE;
  int threadcol = threadIdx.x % BLOCK_COL_SIZE;

  for (int k = 0; k < (K + BK - 1) / BK; k++) {
    const float *A_tile = A_stripe + k * BK;
    const float *B_tile = B_stripe + k * BK * N;
    const int K_remaining = K - k * BK;

    // Threads colletively load the values into the shared memory cache
    for (int tid = threadIdx.x; tid < BM * BK; tid += block_size) {
      int row = tid / BK;
      int col = tid % BK;
      A_tile_cache[row][col] = (row < M_remaining && col < K_remaining)
                                   ? A_tile[row * K + col]
                                   : 0.0;
    }

    // B_tile_cache contains BK*BN values; load the complete tile.
    for (int tid = threadIdx.x; tid < BK * BN; tid += block_size) {
      int row = tid / BN;
      int col = tid % BN;
      B_tile_cache[row][col] = (row < K_remaining && col < N_remaining)
                                   ? B_tile[row * N + col]
                                   : 0.0;
    }

    // Ensure that threads in the block all finish the read
    __syncthreads();

    for (int k = 0; k < BK; k++) {
      for (int i = 0; i < TM; i++) {
        TA_tile_cache[i] = A_tile_cache[threadrow * TM + i][k];
      }
      for (int i = 0; i < TN; i++) {
        TB_tile_cache[i] = B_tile_cache[k][threadcol * TN + i];
      }

      for (int row = 0; row < TM; row++) {
        for (int col = 0; col < TN; col++) {
          TC_tile_cache[row][col] += TA_tile_cache[row] * TB_tile_cache[col];
        }
      }
    }

    // Ensure all threads in the block finish reading from the A/B tile cache
    __syncthreads();
  }

  // Writeback
  for (int row = 0; row < TM; row++) {
    for (int col = 0; col < TN; col++) {
      int c_row = threadrow * TM + row;
      int c_col = threadcol * TN + col;
      if (c_row < M_remaining && c_col < N_remaining)
        C_tile[c_row * N + c_col] = TC_tile_cache[row][col];
    }
  }
}
