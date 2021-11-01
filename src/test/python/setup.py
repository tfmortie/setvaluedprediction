#from setuptools import setup, Extension
#from torch.utils import cpp_extension

#setup(name='hsoftmax_cpp',
#      ext_modules=[cpp_extension.CppExtension('hsoftmax_cpp', ['hsoftmax.cpp'])],
#      cmdclass={'build_ext': cpp_extension.BuildExtension})

#setup(name='linear_cpp',
#        ext_modules=[cpp_extension.CppExtension('linear_cpp', ['linear.cpp'])],
#        cmdclass={'build_ext': cpp_extension.BuildExtension})

#setup(name='softmax_cpp',
#        ext_modules=[cpp_extension.CppExtension('softmax_cpp', ['softmax.cpp'])],
#        cmdclass={'build_ext': cpp_extension.BuildExtension})

#setup(name='dict_cpp',
#        ext_modules=[cpp_extension.CppExtension('dict_cpp', ['dict.cpp'])],
#        cmdclass={'build_ext': cpp_extension.BuildExtension})

#setup(name="hsoftmax_cpp",
#        ext_modules=[cpp_extension.CppExtension('hsoftmax_cpp', ['hsoftmax.cpp'])],
#        cmdclass={'build_ext': cpp_extension.BuildExtension})

#setup(name="softmax_cpp",
#        ext_modules=[cpp_extension.CppExtension('softmax_cpp', ['softmax.cpp'])],
#        cmdclass={'build_ext': cpp_extension.BuildExtension})

#from setuptools import setup
#from torch.utils.cpp_extension import BuildExtension, CppExtension
#
#ext_modules = [
#    CppExtension(
#        'softmax_cpp',
#        ['softmax.cpp'],
#        extra_compile_args=['-O3', '-g', '-Werror', '-fopenmp'])
#]
#setup(name='softmax_cpp', ext_modules=ext_modules,
#        cmdclass={'build_ext': BuildExtension})

#from setuptools import setup
#from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension
#
#setup(
#    name='softmax_cpp',
#    ext_modules=[
#        CppExtension('softmax_cpp', [
#            'softmax.cpp'
#        ]),
#    ],
#    cmdclass={
#        'build_ext': BuildExtension
#    }
#)




from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

# SOFTMAX
#ext_modules = [
#    CppExtension(
#        'softmax_cpp',
#        ['./softmax.cpp'],
#        extra_compile_args=['-O3', '-g', '-Werror', '-fopenmp'])
#]
#setup(name='softmax_cpp', ext_modules=ext_modules,
#        cmdclass={'build_ext': BuildExtension})

# HSOFTMAX
ext_modules = [
    CppExtension(
        'hsoftmax_cpp',
        ['../cpp/hsoftmax.cpp'],
        #extra_compile_args=['-O3', '-g', '-fopenmp']) # use this line on linux
        extra_compile_args=['-O3', '-g']) # use this line on mac
]
setup(name='hsoftmax_cpp', ext_modules=ext_modules,
        cmdclass={'build_ext': BuildExtension})
