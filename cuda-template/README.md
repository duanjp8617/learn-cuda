# CUDA project template

A cuda project template that can be copy pasted around.

Configure and build with Ninja:

```sh
cmake -S cmake-cuda-template -B build/cmake-cuda-template -G Ninja
ninja -C build/cmake-cuda-template
```

`compile_commands.json` will be generated at
`build/cmake-cuda-template/compile_commands.json`.

# Setting up a cuda dev environment

In order to use this template, a basic cuda dev environment must be setup on the machine.

Assume that the cuda driver has already been installed and `nvidia-smi` command is available. 

Follow the rest of this section to add cuda toolkit and pytorch.

## Install cuda
Install cuda globally using nvidia's official cuda toolkit package. Codex suggests the following commands:
```shell
curl -fsSLO https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i ./cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y cuda-toolkit-13-0
```

By default, cuda is installed to /usr/local/cuda, remember to export the following environment variables in .bashrc:
```shell
# ----------------->Added for the global cuda toolkit installation
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
# <----------------
```

Also, if we install cuda-tookit-13-0, then we'd better install torch from cu130 whell. See later sections.

## Install UV through

```shell
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Initialize UV venv

```shell
mkdir test
cd test
# Initialize a virtual environment at the local workspace
# with python version 3.12
uv venv --python 3.12
# Activate the local venv environment
source .venv/bin/activate
```

## Use uv to manage python packages
```shell
uv pip install torch torchvision torchaudio --index-url https://mirrors.nju.edu.cn/pytorch/whl/cu130/
```

## Test code

Now test with this matadd example to see if everything works. It uses torch extention to compile the add.cu into a pytorch module and then loads the module from the main.py

```cpp
#include <torch/extension.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)                                                                                                 \
  CHECK_CUDA(x);                                                                                                       \
  CHECK_CONTIGUOUS(x)

__global__ void add_kernel(const float *input1, const float *input2, float *output, int size) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size)
    output[idx] = input1[idx] + input2[idx];
}

torch::Tensor add(torch::Tensor input1, torch::Tensor input2) {
  CHECK_INPUT(input1);
  CHECK_INPUT(input2);
  int size = input1.numel();
  TORCH_CHECK(size == input2.numel(), "input1 and input2 must have the same size");
  torch::Tensor output = torch::empty(size, input1.options());

  int n_threads = 256;
  int n_blocks = (size + n_threads - 1) / n_threads;
  add_kernel<<<n_blocks, n_threads>>>(input1.data_ptr<float>(), input2.data_ptr<float>(), output.data_ptr<float>(),
                                      size);

  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("add", &add, "Add two vectors"); }
```

```python
import torch
import torch.utils.cpp_extension

module = torch.utils.cpp_extension.load(
    "module",
    sources=["add.cu"],
    extra_cuda_cflags=["-O3"],
    verbose=True,
)

# Example usage
input1 = torch.randn(1000, device="cuda")
input2 = torch.randn(1000, device="cuda")
output = module.add(input1, input2)

torch.testing.assert_close(output, input1 + input2)
```