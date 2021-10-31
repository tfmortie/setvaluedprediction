#include <torch/extension.h>
#include <Python.h>
#include <iostream>
#include <vector>

//https://stackoverflow.com/questions/49988922/how-to-convert-a-returned-python-dictionary-to-a-c-stdmapstring-string

torch::Tensor dict_process(
        torch::Tensor input,
        PyObject* tree
        ) {
    return input;

};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dict_process", &dict_process, "Test process");
}
