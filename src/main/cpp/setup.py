from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

ext_modules = [
    CppExtension(
        'svbop_cpp',
        ['svbop.cpp'],
        #extra_compile_args=['-O3', '-g', '-fopenmp']) # use this line on linux
        extra_compile_args=['-O3', '-g']) # use this line on mac
]
setup(name='svbop_cpp', ext_modules=ext_modules,
        cmdclass={'build_ext': BuildExtension})