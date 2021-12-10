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
        Hierarchical structure of the classification problem. If none,
        a flat probabilistic model is going to be considered.
    
    Attributes
    ----------
    """
    def __init__(self, phi, hidden_size, num_classes, hstruct=None):
        super(SVPNet, self).__init__()
        self.phi = phi
        self.flatten = nn.Flatten()
        self.SVP = SVP(hidden_size, num_classes, hstruct)

    def forward(self, x, y):
        """ Forward pass for the set-valued predictor.
        
        Parameters
        ----------
        x : Input tensor of size (N, D) 
            Represents a batch.
        y : Nested list of int
            Represents the target labels.

        Returns
        -------
        o : Loss tensor of size (N,)
        """
        o = self.phi(x)
        o = self.flatten(o)
        o = self.SVP(o, y)
        
        return o

    def predict(self, x):
        """ Predict function for the set-valued predictor.
        
        Parameters
        ----------
        x : Input tensor of size (N, D) 
            Represents a batch.

        Returns
        -------
        y : Output tensor of size (N,)
        """

        return "Not implemented yet!"
    
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
