"""
Implementation for PyTorch set-valued predictor.

Author: Thomas Mortier
Date: November 2021

TODO:
    - improve checks for param in predict_set
"""
import torch
import torch.nn as nn
import numpy as np
from svp_cpp import SVP

class SVPNet(torch.nn.Module):
    """ Pytorch module which represents a set-valued predictor.

    Parameters
    ----------
    phi : torch.nn.Module
        Represents the neural network architecture which learns the hidden representations
        for the probabilistic model. Must be of type torch.nn.Module with output 
        (batch_size, hidden_size).
    hidden_size : int
        Size of the hidden representation which is passed to the probabilistic model.
    num_classes : int 
        Number of classes.
    hstruct : nested list of int or tensor, default=None
        Hierarchical structure of the classification problem. If None,
        a flat probabilistic model is going to be considered.
    transformer : SVPTransformer, default=None
        Transformer needed for the SVP module. Is None in case of a flat
        probabilistic model.

    Attributes
    ----------
    phi : torch.nn.Module
        Represents the neural network architecture which learns the hidden representation
        for the probabilistic model. Must be of type torch.nn.Module with output 
        (batch_size, hidden_size).
    hidden_size : int
        Size of the hidden representation which is passed to the probabilistic model.
    num_classes : int 
        Number of classes.
    hstruct : nested list of int or tensor, default=None
        Hierarchical structure of the classification problem. If None,
        a flat probabilistic model is going to be considered.
    transformer : SVPTransformer, default=None
        Transformer needed for the SVP module. Is None in case of a flat
        probabilistic model.
    SVP : SVP module
        SVP module.
    """
    def __init__(self, phi, hidden_size, num_classes, hstruct=None, transformer=None):
        super(SVPNet, self).__init__()
        self.phi = phi
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.hstruct = hstruct
        self.transformer = transformer
        if hstruct is None:
            self.SVP = SVP(self.hidden_size, self.num_classes, [])
        else:
            self.SVP = SVP(self.hidden_size, self.num_classes, self.hstruct)

    def forward(self, x, y):
        """ Forward pass for the set-valued predictor.
        
        Parameters
        ----------
        x : input tensor of size (N, D) 
            Represents a batch.
        y : target tensor
            Represents the target labels.

        Returns
        -------
        o : loss tensor of size (N,)
        """
        # get embeddings
        x = self.phi(x)
        # inverse transform labels
        if type(self.hstruct) is torch.Tensor:
            y = self.transformer.transform(y.tolist(), False)
            y = torch.Tensor(sum(y,[])).long().to(x.device)
        else:
            y = self.transformer.transform(y.tolist(), True)
        o = self.SVP(x, y)

        return o

    def predict(self, x):
        """ Predict function for the set-valued predictor.
        
        Parameters
        ----------
        x : input tensor of size (N, D) 
            Represents a batch.

        Returns
        -------
        o : output list of size (N,)
        """
        # get embeddings
        x = self.phi(x)
        # get top-1 predictions
        o = self.SVP.predict(x) 
        o = self.transformer.inverse_transform(o)

        return o
    
    def predict_set(self, x, params):
        """ Predict set function for the set-valued predictor 
        
        Parameters
        ----------
        x : input tensor of size (N, D) 
            Represents a batch.
        params : dict
            Represents parameters for the set-valued prediction task.

        Returns
        -------
        o : nested output list of size (N,)
        """
        # get embeddings
        x = self.phi(x)
        # process params and get set
        try:
            c = int(params["c"])
        except ValueError:
            raise ValueError("Invalid representation complexity {0}. Must be integer.".format(params["c"]))
        if params["constraint"] == "none":
            try:
                beta = int(params["beta"])
            except ValueError:
                raise ValueError("Invalid beta {0}. Must be positive integer.".format(params["beta"]))
            o_t = self.SVP.predict_set_fb(x, beta, c)
        elif params["constraint"] == "size":
            try:
                size = int(params["size"])
            except ValueError:
                raise ValueError("Invalid size {0}. Must be positive integer.".format(params["size"]))
            o_t = self.SVP.predict_set_size(x, size, c)
        elif params["constraint"] == "error":
            try:
                error = float(params["error"])
            except ValueError:
                raise ValueError("Invalid error {0}. Must be a real number in [0,1].".format(params["error"]))
            o_t = self.SVP.predict_set_error(x, error, c)
        else: 
            raise ValueError("Invalid constraint {0}! Valid options: {none, size, error}.".format(params["constraint"]))
        # inverse transform sets
        o = [] 
        for o_t_i in o_t:
            o_t_i = self.transformer.inverse_transform(o_t_i)
            o.append(o_t_i)

        return o
