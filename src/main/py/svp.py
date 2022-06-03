"""
Implementation for PyTorch and Scikit-learn set-valued predictors.

Author: Thomas Mortier
Date: November 2021

TODO:
    - improve checks for param in predict_set
    - argument checks
    - information in predict_set related to different settings
"""
import torch

from svp_cpp import SVP
from .utils import LabelTransformer

class SVPNet(torch.nn.Module):
    """ Pytorch module which represents a set-valued predictor.

    Parameters
    ----------
    phi : torch.nn.Module
        Represents the neural network architecture which returns the hidden representations
        for the probabilistic model. Must be of type torch.nn.Module with output 
        (batch_size, hidden_size).
    hidden_size : int
        Size of the hidden representation which is passed to the probabilistic model.
    classes : list
        List containing classes seen during training time. 
    hierarchy : {'predefined', 'random', 'none'}, default='none'
        Type of probabilistic model to consider in the set-valued predictor.
    k : tuple of int, default=None
        Min and max number of children a node can have in the random generated tree. Is only used when hierarchy='random'. 
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
    classes_ : list
        List containing classes seen during training time. 
    hierarchy : {'predefined', 'random', 'none'}, default='none'
        Type of probabilistic model to consider in the set-valued predictor.
    k : tuple of int, default=None
        Min and max number of children a node can have in the random generated tree. Is only used when hierarchy='random'. 
    random_state : RandomState or an int seed, default=None
        A random number generator instance to define the state of the
        random generator.
    transformer : LabelTransformer
        Label transformer needed for the SVP module. 
    SVP : SVP module
        SVP module.
    """
    def __init__(self, phi, hidden_size, classes, hierarchy="none", k=None, random_state=None):
        super(SVPNet, self).__init__()
        self.phi = phi
        self.hidden_size = hidden_size
        self.hierarchy = hierarchy
        if self.hierarchy not in ["predefined", "random", "none"]:
            raise ValueError("Argument hierarchy must be in {'predefined', 'random', 'none'}!")
        self.k = k
        self.random_state = random_state
        # create label transfomer
        self.transformer = LabelTransformer(self.hierarchy, self.k, random_state)
        # fit transformer
        self.transformer.fit(classes)
        self.classes_ = self.transformer.hlt.classes_
        if self.hierarchy == "none":
            self.SVP = SVP(self.hidden_size, len(self.classes_), [])
        else:
            self.SVP = SVP(self.hidden_size, len(self.classes_), self.transformer.hstruct_)

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
        o : tensor of size (N,) or (N, K) 
            Returns tensor of loss values if y is not None, and tensor with probabilities of size (N, K) otherwise. Probabilities are sorted wrt self.classes_.
        """
        # get embeddings
        x = self.phi(x)
        if y is not None:
            if type(y) is torch.Tensor:
                y = y.tolist()
            # inverse transform labels
            if self.hierarchy != "none":
                y = self.transformer.transform(y, True)
            else:
                y = torch.Tensor(self.transformer.transform(y, False)).long().to(x.device)
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
        o : list of size (N,)
            Returns output list of predicted classes.
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

            TODO: information related to different settings

        Returns
        -------
        o : nested list of size (N,)
            Returns output list of set-valued predictions.
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