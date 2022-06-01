"""
Implementation for PyTorch and scikit-learn set-valued predictors.

Author: Thomas Mortier
Date: November 2021

TODO:
    - improve checks for param in predict_set
"""
import torch

from svp_cpp import SVP
from .utils import LabelTransformer

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
    classes : list
        List containing classes seen during training time. 
    sep : str, default=None
        Path separator used for processing the hierarchical labels. If set to None,
        a random hierarchy is created and provided flat labels are converted,
        accordingly.
    k : tuple of int, default=None
        Min and max number of children a node can have in the random generated tree. Is ignored when
        sep is not set to None.
    random_state : RandomState or an int seed, default=None
        A random number generator instance to define the state of the
        random generator.

    Attributes
    ----------
    phi : torch.nn.Module
        Represents the neural network architecture which learns the hidden representations
        for the probabilistic model. Must be of type torch.nn.Module with output 
        (batch_size, hidden_size).
    hidden_size : int
        Size of the hidden representation which is passed to the probabilistic model.
    classes : list
        List containing classes seen during training time. 
    sep : str, default=None
        Path separator used for processing the hierarchical labels. If set to None,
        a random hierarchy is created and provided flat labels are converted,
        accordingly.
    k : tuple of int, default=None
        Min and max number of children a node can have in the random generated tree. Is ignored when
        sep is not set to None.
    random_state : RandomState or an int seed, default=None
        A random number generator instance to define the state of the
        random generator.
    transformer : LabelTransformer
        Label transformer needed for the SVP module. 
    SVP : SVP module
        SVP module.
    """
    def __init__(self, phi, hidden_size, classes, sep=None, k=None, random_state=None):
        super(SVPNet, self).__init__()
        self.phi = phi
        self.hidden_size = hidden_size
        self.classes = classes
        self.sep = sep
        self.k = k
        self.random_state = random_state
        # create label transfomer
        self.transformer = LabelTransformer(self.k, self.sep, random_state)
        # fit transformer
        self.transformer.fit(self.classes)
        if self.sep is None:
            self.SVP = SVP(self.hidden_size, len(self.transformer.classes_), [])
        else:
            self.SVP = SVP(self.hidden_size, len(self.transformer.classes_), self.transformer.hstruct_)

    def forward(self, x, y=None):
        """ Forward pass for the set-valued predictor.
        
        Parameters
        ----------
        x : input tensor of size (N, D) 
            Represents a batch.
        y : target tensor or list, default=None
            Represents the target labels.

        Returns
        -------
        o : loss tensor of size (N,)
        """
        # get embeddings
        x = self.phi(x)
        if y is not None:
            if type(y) is torch.Tensor:
                y = y.tolist()
            # inverse transform labels
            if self.sep is not None:
                y = self.transformer.transform(y, True)
            else:
                y = self.transformer.transform(y, False)
                y = torch.Tensor(sum(y,[])).long().to(x.device)
            o = self.SVP(x, y)
        else:
            o = self.SVP(x)

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

    def predict_proba(self, x):
        return "Not implemented yet!"
    
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