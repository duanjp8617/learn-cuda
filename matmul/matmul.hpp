#pragma once

template<int M, int N, int K, int BM, int BN, int BK>
__global__ void matmul_1d_tiling(const float* A, const float* B, float* C);

__global__ void matmul_2d_tiling(const float *A, const float *B, float *C);