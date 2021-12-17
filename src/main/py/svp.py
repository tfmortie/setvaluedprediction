"""
Implementation for PyTorch set-valued predictor.

Author: Thomas Mortier
Date: November 2021
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
        Represents the neural network architecture which learns the hidden representation
        for the probabilistic model. Must be of type torch.nn.Module with output 
        (batch_size, hidden_size).
    hidden_size : int
        Size of the hidden representation which is passed to the probabilistic model.
    num_classes : int 
        Number of classes.
    hstruct : nested list of int, default=None
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
    hstruct : nested list of int, default=None
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
        self.SVP = SVP(self.hidden_size, self.num_classes, self.hstruct)

    def forward(self, x, y):
        """ Forward pass for the set-valued predictor.
        
        Parameters
        ----------
        x : Input tensor of size (N, D) 
            Represents a batch.
        y : Nested list of int or Torch tensor
            Represents the target labels.

        Returns
        -------
        o : Loss tensor of size (N,)
        """
        x = self.phi(x)
        if self.transformer is not None:
            y = self.transfomer.transform(y, path=True)
        o = self.SVP(x, y) 

        return o

    def predict(self, x):
        """ Predict function for the set-valued predictor.
        
        Parameters
        ----------
        x : Input tensor of size (N, D) 
            Represents a batch.

        Returns
        -------
        o : Output tensor of size (N,)
        """
        x = self.phi(x)
        o = self.SVP.predict(x) 
        if self.transformer is not None:
            o = self.transformer.inverse_transform(o, path=True)
            o = torch.tensor(o).to(x.device)

        return o
    
    def predict_set(self, x):
        """ Predict set function for the set-valued predictor 
        
        Parameters
        ----------
        x : Input tensor of size (N, D) 
            Represents a batch.

        Returns
        -------
        y : Output tensor of size (N,)
        """

        return "Not implemented yet!"
