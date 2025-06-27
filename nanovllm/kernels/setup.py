import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Get the directory of the setup.py file to build absolute paths
# This is necessary for the compilation to work regardless of the current working directory
setup_dir = os.path.dirname(os.path.abspath(__file__))
kernels_path = os.path.join(setup_dir, 'attention_kernels.cu')
binding_path = os.path.join(setup_dir, 'binding.cpp')

setup(
    name='custom_attention_kernels',
    ext_modules=[
        CUDAExtension(
            name='custom_attention_kernels',
            sources=[
                binding_path,
                kernels_path,
            ],
            # Add any extra compile args if needed for performance tuning
            extra_compile_args={
                'cxx': ['-g', '-O3'],
                'nvcc': ['-O3', '-U__CUDA_NO_HALF_OPERATORS__', '-U__CUDA_NO_HALF_CONVERSIONS__']
            }
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })