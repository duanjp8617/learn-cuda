# CuTe layout printing example

This project is based on `cuda-template` and uses the CUTLASS checkout in
`../cutlass`. It prints the hierarchical layout from the CUTLASS
[CuTe Layout tutorial](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/01_layout.html).

Configure, build, and run:

```sh
mkdir build
cd build
cmake -G Ninja ..
ninja
```

The default preset uses Ninja and creates the `build` directory. If configuring
manually instead, select Ninja explicitly with `cmake -S . -B build -G Ninja`.

To use a CUTLASS checkout in another location, configure with
`-DCUTLASS_DIR=/path/to/cutlass`.
