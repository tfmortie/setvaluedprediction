#include <torch/extension.h>
#include <iostream>
#include <vector>

torch::Tensor linear_forward(
        torch::Tensor input,
        torch::Tensor weights,
        torch::Tensor bias) {
    auto outputs = torch::mm(input, weights);
    bias = bias.unsqueeze(0).expand_as(outputs);
    outputs += bias;

    return outputs;
}

torch::autograd::tensor_list linear_backward(
        torch::Tensor grad_output,
        torch::Tensor input,
        torch::Tensor weights,
        torch::Tensor bias) {
    auto grad_input = torch::mm(grad_output,weights.t());
    auto grad_weights = torch::mm(input.t(),grad_output);
    auto grad_bias = grad_output.sum(0);

    return {grad_input, grad_weights, grad_bias};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &linear_forward, "Linear forward");
    m.def("backward", &linear_backward, "Linear backward");
}
