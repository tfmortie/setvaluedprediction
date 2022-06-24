#!/usr/bin/env python
"""
Setup file for setvaluedprediction package.

Author: Thomas Mortier
Date: June 2022
"""
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

ext_modules = [
    CppExtension(
        'svp_cpp',
        ['svp/svp_cpp.cpp'],
        extra_compile_args=['-O3', '-g', '-fopenmp'],
        extra_link_args=['-lgomp']) # linux
]
setup(name='setvaluedprediction',
      version='0.0.1',
      description='Set-valued predictors in Python',
      author='Thomas Mortier',
      author_email='thomasf.mortier@ugent.be',
      url='https://github.com/tfmortie/setvaluedprediction',
      packages=['svp'],
      install_requires=[
        'joblib == 1.1.0',
        'numpy == 1.22.0',
        'pandas == 1.2.4',
        'scikit-learn == 1.1.1',
        'setuptools == 61.2.0',
        'torch == 1.10.0',
      ],
      ext_modules = ext_modules,
      cmdclass={'build_ext': BuildExtension}
     )
