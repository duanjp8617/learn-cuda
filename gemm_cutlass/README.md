# CuTe SGEMM example

This project builds `sgemm_1.cu`, the first GEMM example from the CUTLASS CuTe
tutorial. The source was moved here from
`cutlass/examples/cute/tutorial/sgemm_1.cu` so it can be built independently
while still using the CUTLASS checkout in `../cutlass` for headers.

## Requirements

- CMake 3.18 or newer
- NVIDIA CUDA toolkit with `nvcc`
- A CUDA-capable GPU
- A CUTLASS checkout at `../cutlass` (the default), or pass another path with
  `-DCUTLASS_DIR=/path/to/cutlass`

Ninja is optional but is used in the commands below.

## Build and run

From the repository root:

```sh
cd gemm_cutlass
cmake -S . -B build -G Ninja
cmake --build build
./build/sgemm_1
```

The defaults are a `5120 x 5120 x 4096` NT GEMM. Pass `M`, `N`, `K`, and the
transpose modes as positional arguments:

```sh
./build/sgemm_1 [M] [N] [K] [transA] [transB]
./build/sgemm_1 1024 1024 1024 N T
```

Only the `NT` and `TN` modes are implemented. To target a specific GPU
architecture instead of automatically detecting one with `nvidia-smi`, pass
`-DCMAKE_CUDA_ARCHITECTURES=<arch>` when configuring, for example `89` for an
Ada GPU.
