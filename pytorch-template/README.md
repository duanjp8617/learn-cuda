# PyTorch CUDA extension template

This directory demonstrates how to call a custom CUDA kernel from PyTorch. The
CUDA/C++ extension is compiled with `torch.utils.cpp_extension`, accepts CUDA
PyTorch tensors, and exposes a vector-add operation to Python.

## Source files

- `runner.py` — Creates two CUDA tensors, calls `vector_add.add`, and verifies
  the result against PyTorch's built-in addition.
- `setup.py` — Defines the `vector_add` CUDA extension and its `nvcc` build
  options.
- `kernel_launcher.cu` — PyTorch/C++ bridge. It validates the inputs,
  allocates the output tensor, selects launch parameters, launches the CUDA
  kernel, and binds the `add` function to Python.
- `vector_add.h` — Templated `__global__` CUDA kernel that performs elementwise
  vector addition.
- `kernel_traits.h` — Compile-time launch traits, including the kernel block
  size.

## Build and run

From this directory, use the project's virtual environment and pytorch will build the
extension directly into `build/`:

```bash
cd /home/djp/learn-cuda/pytorch-template
python setup.py build_ext
python runner.py
```

`runner.py` adds the local `build/` directory to `sys.path`, so no additional
`PYTHONPATH` setting is needed. A successful run prints:

```text
done
```

The assertion in `runner.py` is the result check; it raises an exception if
the custom kernel output differs from `input1 + input2`.
