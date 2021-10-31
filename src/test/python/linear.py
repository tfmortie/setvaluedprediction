import math
import torch

import linear_cpp

class LinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weights, bias):
        outputs = linear_cpp.forward(input, weights, bias)
        variables = [input, weights, bias]
        ctx.save_for_backward(*variables)
        
        return outputs

    @staticmethod
    def backward(ctx, grad_output):
        outputs = linear_cpp.backward(grad_output, *ctx.saved_tensors)
        grad_input, grad_weights, grad_bias = outputs
        
        return grad_input, grad_weights, grad_bias

class Linear(torch.nn.Module):
    def __init__(self, input_features):
        super(Linear, self).__init__()
        self.input_features = input_features
        self.weights = torch.nn.Parameter(
            torch.empty(input_features, 1))
        self.bias = torch.nn.Parameter(torch.empty(1))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0
        for weight in self.parameters():
            weight.data.uniform_(-stdv, +stdv)

    def forward(self, input):
        return LinearFunction.apply(input, self.weights, self.bias)


def testlinear():
    num_features = 5
    model = Linear(num_features)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    X = torch.randn(16,num_features)
    labels = torch.randn(16,1)
    outputs = model(X)
    loss = (outputs-labels).pow(2).sum()
    print(f'{loss=}')
    loss.backward()
    optimizer.step()
    print("Done!")


if __name__=="__main__":
    testlinear()
