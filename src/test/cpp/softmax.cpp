#include <torch/torch.h>
#include <torch/python.h>
#include <torch/extension.h>
#include <iostream>
#include <queue>
#include <sstream>
#include <string>
#include <chrono>
#include <iostream>
#include <algorithm>
#include <random>

struct Softmax : torch::nn::Module {
  // constructor
  Softmax(int64_t D, int64_t K)
      : linear(register_module("linear", torch::nn::Linear(D, K))) {};
  // forward 
  torch::Tensor forward(torch::Tensor input) {
    auto o = linear(input);
    o = torch::nn::functional::softmax(o, torch::nn::functional::SoftmaxFuncOptions(1));
    return o;
  }
  // top-k
  torch::Tensor topk(torch::Tensor input, int64_t k) {
    auto o = linear(input);
    o = torch::nn::functional::softmax(o, torch::nn::functional::SoftmaxFuncOptions(1)); 
    torch::Tensor ind = torch::argsort(o, 1);
    ind = ind.slice(1,0,k,1);
    o = o.gather(1,ind);
    return o;
  }
  // attributes
  torch::nn::Linear linear;
};

PYBIND11_MODULE(softmax_cpp, m) {
   torch::python::bind_module<Softmax>(m, "Softmax")
     .def(py::init<int, int>())
     .def("forward", &Softmax::forward)
     .def("topk", &Softmax::topk);
}