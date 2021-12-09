"""
Setup file for C++ SVP module.

Author: Thomas Mortier
Date: November 2021
"""
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

ext_modules = [
    CppExtension(
        'svp',
        ['hsoftmax_svp.cpp'],
        extra_compile_args=['-O3', '-g', '-fopenmp'],
        extra_link_args=['-lgomp']) # use this line on linux
]
setup(name='svp', ext_modules=ext_modules,
        cmdclass={'build_ext': BuildExtension})
