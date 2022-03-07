"""
Implementation for PyTorch set-valued predictor.

Author: Thomas Mortier
Date: November 2021

TODO:
    - improve checks for param in predict_set
"""
import torch
import cvxpy
import time
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
        if self.hstruct is None:
            self.SVP = SVP(self.hidden_size, self.num_classes, [])
        else:
            self.SVP = SVP(self.hidden_size, self.num_classes, self.hstruct)
        self.transformer = transformer

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
            if type(self.hstruct) is list and len(self.hstruct)>0:
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

    def pwk_ilp_get_ab(self, hstruct, params):
        A = []
        A.append(np.array([len(s) for s in hstruct]))
        # add 1
        A.append(np.ones(len(hstruct)))
        # add E
        # run over adjecency matric
        for i in range(len(hstruct)):
            for j in range(i+1,len(hstruct)):
                if len(set(hstruct[i])&set(hstruct[j]))>0:
                    # we have found an edge
                    e = np.zeros(len(hstruct))
                    e[i] = 1
                    e[j] = 1
                    A.append(e)
        A = np.vstack(A)
        # construct b
        b = np.ones(A.shape[0])
        b[0] = params["size"]
        b[1] = params["c"]

        return A, b
    
    def pwk_ilp(self, P, A, b, hstruct, solver):
        o_t = [] 
        t = 0.0
        for pi in P:
            # get p
            p = []
            for s in hstruct:
                p_s = 0
                for si in s:
                    p_s += pi[si]
                p.append(p_s)
            p = np.array(p)
            # solve our ILP
            start_time = time.time()
            selection = cvxpy.Variable(len(hstruct), boolean=True)
            constraint = A @ selection <= b
            utility = p @ selection
            knapsack_problem = cvxpy.Problem(cvxpy.Maximize(utility), [constraint])
            if solver=="GLPK_MI":
                knapsack_problem.solve(solver=cvxpy.GLPK_MI)
            elif solver=="CBC":
                knapsack_problem.solve(solver=cvxpy.CBC)
            else:
                knapsack_problem.solve(solver=cvxpy.SCIP)
            stop_time = time.time()
            t += stop_time-start_time
            sel_ind = list(np.where(selection.value)[0])
            pred = []
            for i in sel_ind:
                pred.extend(hstruct[i])
            o_t.append(pred)
        t /= P.shape[0]
        
        # inverse transform sets
        o = [] 
        for o_t_i in o_t:
            o_t_i = self.transformer.inverse_transform(o_t_i)
            o.append(o_t_i)
 
        return o, t
    
    def set_hstruct(self, hstruct):
        self.hstruct = hstruct
        if self.hstruct is None:
            self.SVP.set_hstruct([])
        else:
            self.SVP.set_hstruct(hstruct)
