#include "matmul.hpp"

// This file deliberately uses small iterator-like objects instead of hiding the
// whole kernel behind a generic abstraction.  Each object represents one level
// of the matmul tiling hierarchy:
//
//   1. KTileIterator selects one A(M, K) / B(K, N) pair of shared-memory tiles.
//   2. GlobalTileLoadIterator assigns the elements of one shared-memory tile to
//      all threads in the CTA.
//   3. ThreadOutputTileIterator describes the C elements owned by one thread.
//
// The structs contain only integers and pointers, and all methods are
// __device__ functions.  nvcc can therefore inline them into the kernels; they
// are an organization tool rather than a runtime-polymorphic abstraction.

// Iterates along K for the output tile owned by the current CTA.
//
// On every iteration, the three compatible global-memory tiles are:
//
//   A[m0 : m0 + BM, k0 : k0 + BK]
//   B[k0 : k0 + BK, n0 : n0 + BN]
//   C[m0 : m0 + BM, n0 : n0 + BN]
//
// m0 and n0 are fixed for the lifetime of the CTA.  advance() changes only
// k0, moving the iterator to the next partial-product tile.
template <int BM, int BN, int BK>
struct KTileIterator {
  const int M;
  const int N;
  const int K;
  const int m0;
  const int n0;
  int k0;

  __device__ KTileIterator(int M, int N, int K, int m0, int n0)
      : M(M), N(N), K(K), m0(m0), n0(n0), k0(0) {}

  // The last K tile is valid even when it is smaller than BK.
  __device__ bool valid() const { return k0 < K; }

  // These are the real dimensions of a boundary tile.  They may be smaller
  // than the compile-time shared-memory tile dimensions.
  __device__ int valid_rows() const { return M - m0 < BM ? M - m0 : BM; }
  __device__ int valid_cols() const { return N - n0 < BN ? N - n0 : BN; }
  __device__ int valid_inner() const { return K - k0 < BK ? K - k0 : BK; }

  __device__ void advance() { k0 += BK; }
};

// Iterates over the elements one CTA cooperatively copies into a row-major
// shared-memory tile.  Every thread starts at its thread id, then advances by
// the number of CTA threads.  Together, the threads cover every element once.
//
// For example, with a BK x BN B tile, linear element e corresponds to:
//   tile_row = e / BN, tile_col = e % BN.
// The global address is then matrix[(base_row + tile_row) * leading_dimension
//                                    + base_col + tile_col].
template <int TILE_ROWS, int TILE_COLS>
struct GlobalTileLoadIterator {
  const float *matrix;
  float *cache;
  const int base_row;
  const int base_col;
  const int leading_dimension;
  const int valid_rows;
  const int valid_cols;
  const int thread_count;
  int element;

  __device__ GlobalTileLoadIterator(const float *matrix, float *cache,
                                    int base_row, int base_col,
                                    int leading_dimension, int valid_rows,
                                    int valid_cols, int thread_id,
                                    int thread_count)
      : matrix(matrix),
        cache(cache),
        base_row(base_row),
        base_col(base_col),
        leading_dimension(leading_dimension),
        valid_rows(valid_rows),
        valid_cols(valid_cols),
        thread_count(thread_count),
        element(thread_id) {}

  __device__ bool valid() const { return element < TILE_ROWS * TILE_COLS; }

  __device__ int row() const { return element / TILE_COLS; }
  __device__ int col() const { return element % TILE_COLS; }

  // Zero padding lets the compute phase always execute exactly BK iterations.
  // Thus, only this load step needs boundary logic for an incomplete K tile.
  __device__ void load() const {
    const bool in_bounds = row() < valid_rows && col() < valid_cols;
    cache[element] = in_bounds
                         ? matrix[(base_row + row()) * leading_dimension +
                                  base_col + col()]
                         : 0.0F;
  }

  __device__ void advance() { element += thread_count; }
};

// Describes the TM x TN sub-tile of C owned by one thread.  The output tile is
// tiled by a logical grid with thread_columns threads in each row.  The 1D
// kernel uses TM=TN=1; the 2D kernel uses TM=TN=8.
template <int TM, int TN>
struct ThreadOutputTileIterator {
  const int row0;
  const int col0;
  const int valid_rows;
  const int valid_cols;

  __device__ ThreadOutputTileIterator(int thread_id, int thread_columns,
                                      int valid_rows, int valid_cols)
      : row0((thread_id / thread_columns) * TM),
        col0((thread_id % thread_columns) * TN),
        valid_rows(valid_rows),
        valid_cols(valid_cols) {}

  // row(i) and col(j) are coordinates local to the CTA's C tile.
  __device__ int row(int i) const { return row0 + i; }
  __device__ int col(int j) const { return col0 + j; }

  __device__ bool in_bounds(int i, int j) const {
    return row(i) < valid_rows && col(j) < valid_cols;
  }
};

// Iterator-based version of matmul_1d_tiling.  A thread owns one C value, so
// ThreadOutputTileIterator<1, 1> is the simple one-element microtile case.
template <int M, int N, int K, int BM, int BN, int BK>
__global__ void matmul_1d_tiling_iterator(const float *A, const float *B,
                                          float *C) {
  __shared__ float A_tile_cache[BM][BK];
  __shared__ float B_tile_cache[BK][BN];

  // This CTA owns C[m0:m0+BM, n0:n0+BN].
  KTileIterator<BM, BN, BK> k_tiles(M, N, K, blockIdx.y * BM,
                                    blockIdx.x * BN);
  ThreadOutputTileIterator<1, 1> output_tile(
      threadIdx.x, BN, k_tiles.valid_rows(), k_tiles.valid_cols());

  float accum = 0.0F;

  // Move through K one BK-wide pair of A/B tiles at a time.
  for (; k_tiles.valid(); k_tiles.advance()) {
    GlobalTileLoadIterator<BM, BK> A_loader(
        A, &A_tile_cache[0][0], k_tiles.m0, k_tiles.k0, K,
        k_tiles.valid_rows(), k_tiles.valid_inner(), threadIdx.x, blockDim.x);
    for (; A_loader.valid(); A_loader.advance()) {
      A_loader.load();
    }

    GlobalTileLoadIterator<BK, BN> B_loader(
        B, &B_tile_cache[0][0], k_tiles.k0, k_tiles.n0, N,
        k_tiles.valid_inner(), k_tiles.valid_cols(), threadIdx.x, blockDim.x);
    for (; B_loader.valid(); B_loader.advance()) {
      B_loader.load();
    }

    // All values must be in shared memory before any thread starts reading.
    __syncthreads();

    // A row of A_tile_cache and a column of B_tile_cache form one dot product.
    for (int inner = 0; inner < BK; ++inner) {
      accum += A_tile_cache[output_tile.row(0)][inner] *
               B_tile_cache[inner][output_tile.col(0)];
    }

    // No thread may overwrite shared memory for the next K tile while another
    // thread is still consuming the present one.
    __syncthreads();
  }

  if (output_tile.in_bounds(0, 0)) {
    C[(k_tiles.m0 + output_tile.row(0)) * N +
      k_tiles.n0 + output_tile.col(0)] = accum;
  }
}

template __global__ void matmul_1d_tiling_iterator<37, 29, 53, 16, 16, 16>(
    const float *, const float *, float *);
template __global__ void
matmul_1d_tiling_iterator<1024, 1024, 1024, 16, 16, 16>(const float *,
                                                          const float *, float *);

// Iterator-based version of matmul_2d_tiling.  Unlike the 1D kernel, each
// thread owns an 8 x 8 C microtile.  This reuses each shared-memory value for
// several FMAs and increases the work done per thread.
__global__ void matmul_2d_tiling_iterator(int M, int N, int K, const float *A,
                                          const float *B, float *C) {
  constexpr int BM = 128;
  constexpr int BN = 128;
  constexpr int BK = 32;
  constexpr int TM = 8;
  constexpr int TN = 8;
  constexpr int THREAD_COLUMNS = BN / TN;

  static_assert(BM % TM == 0, "BM must divide evenly into thread microtiles");
  static_assert(BN % TN == 0, "BN must divide evenly into thread microtiles");

  __shared__ float A_tile_cache[BM][BK];
  __shared__ float B_tile_cache[BK][BN];

  KTileIterator<BM, BN, BK> k_tiles(M, N, K, blockIdx.y * BM,
                                    blockIdx.x * BN);
  ThreadOutputTileIterator<TM, TN> output_tile(
      threadIdx.x, THREAD_COLUMNS, k_tiles.valid_rows(), k_tiles.valid_cols());

  // These arrays are registers.  Loading them before the outer-product loop
  // avoids repeatedly reading the same shared-memory values for this thread's
  // TM x TN microtile.
  float A_register[TM];
  float B_register[TN];
  float C_register[TM][TN] = {0.0F};

  for (; k_tiles.valid(); k_tiles.advance()) {
    GlobalTileLoadIterator<BM, BK> A_loader(
        A, &A_tile_cache[0][0], k_tiles.m0, k_tiles.k0, K,
        k_tiles.valid_rows(), k_tiles.valid_inner(), threadIdx.x, blockDim.x);
    for (; A_loader.valid(); A_loader.advance()) {
      A_loader.load();
    }

    GlobalTileLoadIterator<BK, BN> B_loader(
        B, &B_tile_cache[0][0], k_tiles.k0, k_tiles.n0, N,
        k_tiles.valid_inner(), k_tiles.valid_cols(), threadIdx.x, blockDim.x);
    for (; B_loader.valid(); B_loader.advance()) {
      B_loader.load();
    }

    __syncthreads();

    for (int inner = 0; inner < BK; ++inner) {
      // Fetch one A vector and one B vector for the current K coordinate.
      for (int row = 0; row < TM; ++row) {
        A_register[row] = A_tile_cache[output_tile.row(row)][inner];
      }
      for (int col = 0; col < TN; ++col) {
        B_register[col] = B_tile_cache[inner][output_tile.col(col)];
      }

      // Their outer product contributes to all TM x TN accumulators.
      for (int row = 0; row < TM; ++row) {
        for (int col = 0; col < TN; ++col) {
          C_register[row][col] += A_register[row] * B_register[col];
        }
      }
    }

    __syncthreads();
  }

  // The load iterators padded edge tiles with zero, but stores still need a
  // bounds check so a boundary CTA never writes beyond C.
  for (int row = 0; row < TM; ++row) {
    for (int col = 0; col < TN; ++col) {
      if (output_tile.in_bounds(row, col)) {
        C[(k_tiles.m0 + output_tile.row(row)) * N +
          k_tiles.n0 + output_tile.col(col)] = C_register[row][col];
      }
    }
  }
}
