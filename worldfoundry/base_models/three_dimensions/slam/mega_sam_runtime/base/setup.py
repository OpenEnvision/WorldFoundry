"""Build the two CUDA extensions required by the canonical MegaSAM runtime."""

from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT = Path(__file__).resolve().parent

setup(
    name="worldfoundry-megasam-extensions",
    version="0.2",
    packages=["lietorch"],
    package_dir={"": "thirdparty/lietorch"},
    ext_modules=[
        CUDAExtension(
            "droid_backends",
            include_dirs=[str(ROOT / "thirdparty/eigen")],
            sources=["src/droid.cpp", "src/droid_kernels.cu", "src/correlation_kernels.cu", "src/altcorr_kernel.cu"],
            extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3"]},
        ),
        CUDAExtension(
            "lietorch_backends",
            include_dirs=[str(ROOT / "thirdparty/lietorch/lietorch/include"), str(ROOT / "thirdparty/eigen")],
            sources=[
                "thirdparty/lietorch/lietorch/src/lietorch.cpp",
                "thirdparty/lietorch/lietorch/src/lietorch_gpu.cu",
                "thirdparty/lietorch/lietorch/src/lietorch_cpu.cpp",
            ],
            extra_compile_args={"cxx": ["-O2"], "nvcc": ["-O2"]},
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
