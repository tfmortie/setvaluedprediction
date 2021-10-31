import math
import torch

import dict_cpp

class DictFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weights, bias):


    @staticmethod
    def backward(ctx, grad_output):



class Dict(torch.nn.Module):
    def __init__(self, input_features):
        super(Dict, self).__init__()
        self.input_features = input_features
        self.tree = nn.ModuleDict({
            "root": nn.ModuleDict({"children": ["


        })
