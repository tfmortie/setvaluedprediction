"""
JIT compilation of C++ extension

TODO: remove JIT compilation in near future -> slow loading times

Author: Thomas Mortier
Date: June 2022
"""
import glob
import os.path

try:
    from torch.utils.cpp_extension import load
except ImportError:
    raise ImportError("Included extensions require PyTorch 0.4 or higher")


def _load_C_extension():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    this_dir = os.path.join(this_dir, "csrc")
    source = glob.glob(os.path.join(this_dir, "*.cpp"))
    source = [os.path.join(this_dir, s) for s in source]
    print("source: {0}".format(source))
    print("this_dir: {0}".format(this_dir))
    extra_include_paths = [this_dir]
    return load("svp_cpp", source, extra_include_paths=extra_include_paths)


_C = _load_C_extension()
