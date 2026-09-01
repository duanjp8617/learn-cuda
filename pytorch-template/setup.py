from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="vector_add",
    ext_modules=[
        CUDAExtension(
            "vector_add",
            [
                "kernel_launcher.cu",                
            ],
            extra_compile_args={
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    "-gencode",
                    "arch=compute_89,code=sm_89",
                    "-std=c++17",
                ],
            },
            verbose=True,
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)