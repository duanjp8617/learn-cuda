# CUDA CMake template

This template follows the tutorial's conventional `include/` and `src/` layout,
uses `find_package(CUDAToolkit)`, source and include-directory variables,
target-based source/link configuration, and CUDA separable compilation.

Configure and build with Ninja:

```sh
cmake -S cmake-cuda-template -B build/cmake-cuda-template -G Ninja
ninja -C build/cmake-cuda-template
```

`compile_commands.json` is generated at
`build/cmake-cuda-template/compile_commands.json`.
