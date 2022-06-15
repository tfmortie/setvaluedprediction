"""
Implementation of PyTorch and Scikit-learn set-valued predictors.

Author: Thomas Mortier
Date: November 2021
"""
import torch
import time
import warnings
import numpy as np

from svp_cpp import SVP
from .utils import LabelTransformer
from .utils import FLabelTransformer, PriorityQueue
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils import _message_with_time
from sklearn.utils.validation import check_X_y, check_array, check_random_state
from sklearn.exceptions import NotFittedError, FitFailedWarning
from sklearn.metrics import accuracy_score

from joblib import Parallel, delayed, parallel_backend
from collections import ChainMap

class SVPNet(torch.nn.Module):
    """ Pytorch module which represents a set-valued predictor.

    Parameters
    ----------
    phi : torch.nn.Module
        Represents the neural network architecture which returns the hidden representations for the probabilistic model. Must be of type torch.nn.Module with output (batch_size, hidden_size).
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
        Represents the neural network architecture which learns the hidden representations. for the probabilistic model. Must be of type torch.nn.Module with output (batch_size, hidden_size).
    hidden_size : int
        Size of the hidden representation which is passed to the probabilistic model.
    hierarchy : {'predefined', 'random', 'none'}, default='none'
        Type of probabilistic model to consider in the set-valued predictor.
    k : tuple of int, default=None
        Min and max number of children a node can have in the random generated tree. Is only used when hierarchy='random'. 
    random_state : RandomState or an int seed, default=None
        A random number generator instance to define the state of the random generator.
    transformer : LabelTransformer
        Label transformer needed for the SVP module. 
    classes_ : list
        List containing classes seen during training time. 
    SVP : SVP module
        SVP module.

    Examples
    --------
    >>> from svp import SVPNet 
    >>> import torch.nn as nn
    >>> 
    >>> net = SVPNet(nn.Linear(1000,1000),
    >>>         hidden_size = 1000,
    >>>         classes = y,
    >>>         hierarchy="random",
    >>>         k=(2,2),
    >>>         random_state=0)
    >>> clf(X)
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
        # register classes 
        if self.hierarchy != "random":
            self.classes_ = self.transformer.hlt.classes_
        else:
            self.classes_ = self.transformer.flt.inverse_transform(self.transformer.hlt.classes_)
        if self.hierarchy == "none":
            self.SVP = SVP(self.hidden_size, len(self.classes_), [])
        else:
            self.SVP = SVP(self.hidden_size, len(self.classes_), self.transformer.hstruct_)

    def forward(self, X, y=None):
        """ Forward pass for the set-valued predictor.
        
        Parameters
        ----------
        X : tensor, shape (n_samples, n_features) 
            The input samples
        y : {tensor, list}, shape (n_samples,), default=None
            The class labels

        Returns
        -------
        o : tensor, shape (n_samples,) or (n_samples, n_classes)
            Returns tensor of loss values if y is not None, and tensor with probabilities of size (n_samples, n_classes) otherwise. Probabilities are sorted wrt self.classes_.
        """
        # get embeddings
        x = self.phi(X)
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

    def predict(self, X):
        """ Predict function for the set-valued predictor.
        
        Parameters
        ----------
        X : tensor, shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        o : list, size (n_samples,)
            Returns output list of predicted classes.
        """
        # get embeddings
        x = self.phi(X)
        # get top-1 predictions
        o = self.SVP.predict(x) 
        o = self.transformer.inverse_transform(o)

        return o
 
    def predict_set(self, X, params):
        """ Return set-valued predictions.
        
        Parameters
        ----------
        X : tensor, shape (n_samples, n_features)
            Input samples.
        params : dict
            Represents parameters for the set-valued prediction task. Must contain following keys:
                - c, int
                    Representation complexity.
                - svptype, str {"fb", "dg", "sizectrl", "errorctrl"}
                    Type of set-valued predictor.
                - beta, int
                    Beta parameter in case of svptype="fb"
                - delta, float
                    Float parameter in case of svptype="dg"
                - gamma, float
                    Float parameter in case of svptype="dg"
                - size, int
                    Size parameter in case of svptype="sizectrl"
                - error, float
                    Error parameter in case of svptype="errorctrl"

        Returns
        -------
        o : list, size (n_samples,)
            Nested list of set-valued predictions.
        """
        # get embeddings
        x = self.phi(X)
        # process params and get set
        try:
            c = int(params["c"])
        except ValueError:
            raise ValueError("Invalid representation complexity {0}. Must be integer.".format(params["c"]))
        if params["svptype"] == "fb":
            try:
                beta = int(params["beta"])
            except ValueError:
                raise ValueError("Invalid beta {0}. Must be positive integer.".format(params["beta"]))
            o_t = self.SVP.predict_set_fb(x, beta, c)
        elif params["svptype"] == "dg":
            try:
                delta = float(params["delta"])
                gamma = float(params["gamma"])
            except ValueError:
                raise ValueError("Invalid delta {0} or gamma {1}. Must be positive float.".format(params["delta"], params["gamma"]))
            o_t = self.SVP.predict_set_dg(x, delta, gamma, c)
        elif params["svptype"] == "sizectrl":
            try:
                size = int(params["size"])
            except ValueError:
                raise ValueError("Invalid size {0}. Must be positive integer.".format(params["size"]))
            o_t = self.SVP.predict_set_size(x, size, c)
        elif params["svptype"] == "errorctrl":
            try:
                error = float(params["error"])
            except ValueError:
                raise ValueError("Invalid error {0}. Must be a real number in [0,1].".format(params["error"]))
            o_t = self.SVP.predict_set_error(x, error, c)
        else: 
            raise ValueError("Invalid SVP type {0}! Valid options: {fb, dg, sizectrl, errorctrl}.".format(params["svptype"]))
        # inverse transform sets
        o = [] 
        for o_t_i in o_t:
            o_t_i = self.transformer.inverse_transform(o_t_i)
            o.append(o_t_i)

        return o

class SVPClassifier(BaseEstimator, ClassifierMixin):
    """ Scikit-learn module which represents a set-valued predictor.

    Parameters
    ----------
    estimator : scikit-learn base estimator
        Represents the base estimator for the classification task in each node (in case of hierarchical model).
    hierarchy : {'predefined', 'random', 'none'}, default='none'
        Type of probabilistic model to consider in the set-valued predictor.
    k : tuple of int, default=None
        Min and max number of children a node can have in the random generated tree. Is only used when hierarchy='random'.
    n_jobs : int, default=1
        The number of jobs to run in parallel.
    random_state : RandomState or an int seed, default=None
        A random number generator instance to define the state of the random generator.
    verbose : int, default=0
        Controls the verbosity: the higher, the more messages

    Attributes
    ----------
    estimator : scikit-learn base estimator
        Represents the base estimator for the classification task in each node (in case of hierarchical model).
    hierarchy : {'predefined', 'random', 'none'}, default='none'
        Type of probabilistic model to consider in the set-valued predictor.
    k : tuple of int, default=None
        Min and max number of children a node can have in the random generated tree. Is only used when hierarchy='random'.
    n_jobs : int, default=1
        The number of jobs to run in parallel.
    verbose : int, default=0
        Controls the verbosity: the higher, the more messages.
    random_state_ : RandomState or an int seed, default=None
        A random number generator instance to define the state of the random generator.
    X_ : {array-like, sparse matrix}, shape (n_samples, n_features)
        Training input samples seen during fit.
    y_ : array-like, shape (n_samples,)
        The class labels seen during fit.
    label_encoder_ : FLabelTransformer
        Label transformer needed for the SVPClassifier module.
    rlbl_ : str
        Label of root node (in case of hierarchical model).
    tree_ : dict
        Represents the fitted tree structure in case of a hierarhical probabilistic model.
    classes_ : list
        List containing classes seen during fit.

    Examples
    --------
    >>> from svp import SVPClassifier 
    >>> from sklearn.linear_model import LogisticRegression
    >>> 
    >>> clf = SVPClassifier(LogisticRegression(random_state=0),
    >>>         hierarchy="random",
    >>>         k=(2,2),
    >>>         n_jobs=4,
    >>>         random_state=0,
    >>>         verbose=1)
    >>> clf.fit(X, y)
    >>> clf.score(X, y)
    """
    def __init__(self, estimator, hierarchy="none", k=None, n_jobs=1, random_state=None, verbose=0):
        self.estimator = clone(estimator)
        self.hierarchy = hierarchy
        self.k = k
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.tree_ = {}

    def _add_path(self, path):
        current_node = path[0]
        add_node = path[1]
        # check if add_node is already registred
        if add_node not in self.tree_:
            # register add_node to the tree
            self.tree_[add_node] = {
                "lbl": add_node,
                "estimator": None,
                "children": [],
                "parent": current_node} 
            # add add_node to current_node's children (if not yet in list of children)
            if add_node not in self.tree_[current_node]["children"]:
                self.tree_[current_node]["children"].append(add_node)
            # set estimator when num. of children for current_node is higher than 1 and if not yet set
            if len(self.tree_[current_node]["children"]) > 1 and self.tree_[current_node]["estimator"] is None:
                self.tree_[current_node]["estimator"] = clone(self.estimator)
        else:
            # check for cycles
            if self.tree_[add_node]["parent"] != current_node and current_node != add_node:
                warnings.warn("Duplicate node label {0} detected in hierarchy with parents {1}, {2}!".format(add_node, self.tree_[add_node]["parent"], current_node), FitFailedWarning)
        # process next couple of nodes in path
        if len(path) > 2:
            path = path[1:]
            self._add_path(path)

    def _fit_node(self, node):
        # check if node has estimator
        if node["estimator"] is not None:
            # transform data for node
            y_transform = []
            sel_ind = []
            for i,y in enumerate(self.y_):
                if node["lbl"] in y.split(";"):
                    # need to include current label and sample (as long as it's "complete")
                    y_split = y.split(";")
                    y_idx = len(y_split)-y_split[::-1].index(node["lbl"])-1
                    if y_idx < len(y_split)-1:
                        y_transform.append(y_split[y_idx+1])
                        sel_ind.append(i)
            X_transform = self.X_[sel_ind,:]
            node["estimator"].fit(X_transform, y_transform)
            if self.verbose >= 2:
                print("Model {0} fitted!".format(node["lbl"]))
            # now make sure that the order of labels correspond to the order of children
            node["children"] = node["estimator"].classes_
        return {node["lbl"]: node}
        
    def fit(self, X, y):
        """ Implementation of the fit function for the set-valued predictor.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,)
            The class labels

        Returns
        -------
        self : object
            Returns self.
        """
        self.random_state_ = check_random_state(self.random_state)
        # need to make sure that X and y have the correct shape
        X, y = check_X_y(X, y, multi_output=False) # multi-output not supported (yet)
        # check if n_jobs is integer
        if not isinstance(self.n_jobs, int):
            raise TypeError("Parameter n_jobs must be of type int.")
        # store data seen during fit
        self.X_ = X
        self.y_ = y
        # check if flat or hierarchical model
        start_time = time.time()
        if self.hierarchy != "none":
            # first init the tree 
            try:
                if self.hierarchy == "random":
                    if self.k is None:
                        self.label_encoder_ = FLabelTransformer(sep=";", k=(2,2), random_state=self.random_state_)
                    else:
                        self.label_encoder_ = FLabelTransformer(sep=";", k=self.k, random_state=self.random_state_)
                    self.y_ = self.label_encoder_.fit_transform(self.y_) 
                else:
                    self.label_encoder_ = None
                # store label of root node
                self.rlbl_ = self.y_[0].split(";")[0]
                # init tree
                self.tree_ = {self.rlbl_: {
                    "lbl": self.rlbl_,
                    "estimator": None,
                    "children": [],
                    "parent": None}}
                for lbl in self.y_:
                    path = lbl.split(";")
                    self._add_path(path)
                # now proceed to fitting
                with parallel_backend("loky"):
                    fitted_tree = Parallel(n_jobs=self.n_jobs)(delayed(self._fit_node)(self.tree_[node]) for node in self.tree_)
                self.tree_ = {k: v for d in fitted_tree for k, v in d.items()}
            except NotFittedError as e:
                raise NotFittedError("Tree fitting failed! Make sure that the provided data is in the correct format.")
            # now store classes (leaf nodes) seen during fit
            cls = []
            nodes_to_visit = [self.tree_[self.rlbl_]]
            while len(nodes_to_visit) > 0:
                curr_node = nodes_to_visit.pop()
                for c in curr_node["children"]:
                    # check if child is leaf node 
                    if len(self.tree_[c]["children"]) == 0:
                        cls.append(c)
                    else:
                        # add child to nodes_to_visit
                        nodes_to_visit.append(self.tree_[c])
            self.classes_ = cls 
            # make sure that classes_ are in same format of original labels
            if self.label_encoder_ is not None:
                self.classes_ = self.label_encoder_.inverse_transform(self.classes_)
            else:
                # construct dict with leaf node lbls -> path mappings
                lbl_to_path = {yi.split(";")[-1]: yi for yi in self.y_}
                self.classes_ = [lbl_to_path[cls] for cls in self.classes_]
        else:
            self.estimator.fit(X, y)
            self.classes_ = self.estimator.classes_
        stop_time = time.time()
        if self.verbose >= 1:
            print(_message_with_time("SVPClassifier", "fitting", stop_time-start_time))
        return self
  
    def _predict(self, i, X, scores):
        preds = []
        if self.hierarchy != "none":
            # run over all samples
            for x in X:
                x = x.reshape(1,-1)
                nodes_to_visit = PriorityQueue()
                nodes_to_visit.push(1.,self.rlbl_)
                pred = None
                while not nodes_to_visit.is_empty():
                    curr_node_prob, curr_node = nodes_to_visit.pop()
                    curr_node_lbl = curr_node.split(";")[-1]
                    curr_node_prob = 1-curr_node_prob
                    # check if we are at a leaf node
                    if len(self.tree_[curr_node_lbl]["children"]) == 0:
                        pred = curr_node
                        break
                    else:
                        curr_node_v = self.tree_[curr_node_lbl]
                        # check if we have a node with single path
                        if curr_node_v["estimator"] is not None:
                            # get probabilities
                            curr_node_ch_probs = self._predict_proba(curr_node_v["estimator"], x, scores)
                            # apply chain rule of probability
                            curr_node_ch_probs = curr_node_ch_probs*curr_node_prob
                            # add children to queue
                            for j,c in enumerate(curr_node_v["children"]):
                                prob_child = curr_node_ch_probs[:,j][0]
                                nodes_to_visit.push(prob_child, curr_node+";"+c)
                        else:
                            c = curr_node_v["children"][0]
                            nodes_to_visit.push(curr_node_prob,curr_node+";"+c)
                preds.append(pred)
        else:
            preds = self.estimator.predict(X)
        return {i: preds}
    
    def predict(self, X):
        """ Predict function for the set-valued predictor.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        o : list, size (n_samples,)
            Returns output list of predicted classes.
        """
        # check input
        X = check_array(X)
        scores = False
        o = []
        start_time = time.time()
        # check whether the base estimator supports probabilities
        if not hasattr(self.estimator, 'predict_proba'):
            # check whether the base estimator supports class scores
            if not hasattr(self.estimator, 'decision_function'):
                raise NotFittedError("{0} does not support \
                         probabilistic predictions nor scores.".format(self.estimator))
            else:
                scores = True
        try:
            # now proceed to predicting
            with parallel_backend("loky"):
                d_preds = Parallel(n_jobs=self.n_jobs)(delayed(self._predict)(i,X[ind],scores) for i,ind in enumerate(np.array_split(range(X.shape[0]), self.n_jobs)))
            # collect predictions
            preds_dict = dict(ChainMap(*d_preds))
            for k in np.sort(list(preds_dict.keys())):
                o.extend(preds_dict[k])
            if self.hierarchy != "none":
                # in case of no predefined hierarchy, backtransform to original labels
                if self.label_encoder_ is not None:
                    o = self.label_encoder_.inverse_transform([p.split(";")[-1] for p in o])
        except NotFittedError as e:
            raise NotFittedError("This model is not fitted yet. Cal 'fit' \
                    with appropriate arguments before using this \
                    method.")
        stop_time = time.time()
        if self.verbose >= 1:
            print(_message_with_time("SVPClassifier", "predicting", stop_time-start_time))

        return o
     
    def _predict_proba(self, estimator, X, scores=False):
        if not scores:
            return estimator.predict_proba(X)
        else:
            # get scores
            scores = estimator.decision_function(X)
            scores = np.exp(scores)
            # check if we only have one score (ie, when K=2)
            if len(scores.shape) == 2:
                # softmax evaluation
                scores = scores/np.sum(scores,axis=1).reshape(scores.shape[0],1)
            else:
                # sigmoid evaluation
                scores = 1/(1+np.exp(-scores))
                scores = scores.reshape(-1,1)
                scores = np.hstack([1-scores,scores])
            return scores
    
    def predict_proba(self, X):
        """ Predict function for the set-valued predictor that returns probability estimates, where classes are ordered by self.classes_.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        o : array-like, shape (n_samples, n_classes)
            Returns the probability of the sample for each class in the model, where classes are ordered by self.classes_.
        """
        # check input
        X = check_array(X)
        scores = False
        o = []
        start_time = time.time()
        # check whether the base estimator supports probabilities
        if not hasattr(self.estimator, 'predict_proba'):
            # check whether the base estimator supports class scores
            if not hasattr(self.estimator, 'decision_function'):
                raise NotFittedError("{0} does not support \
                         probabilistic predictions nor scores.".format(self.estimator))
            else:
                scores = True
        if self.hierarchy != "none":
            try:
                nodes_to_visit = [(self.tree_[self.rlbl_], np.ones((X.shape[0],1)))]
                while len(nodes_to_visit) > 0:
                    curr_node, parent_prob = nodes_to_visit.pop()
                    # check if we have a node with single path
                    if curr_node["estimator"] is not None:
                        # get probabilities 
                        curr_node_probs = self._predict_proba(curr_node["estimator"], X, scores)
                        # apply chain rule of probability
                        curr_node_probs = curr_node_probs*parent_prob
                        for i,c in enumerate(curr_node["children"]):
                            # check if child is leaf node 
                            prob_child = curr_node_probs[:,i].reshape(-1,1)
                            if len(self.tree_[c]["children"]) == 0:
                                o.append(prob_child)
                            else:
                                # add child to nodes_to_visit
                                nodes_to_visit.append((self.tree_[c], prob_child))
                    else:
                        c = curr_node["children"][0]
                        # check if child is leaf node 
                        if len(self.tree_[c]["children"]) == 0:
                            o.append(parent_prob)
                        else:
                            # add child to nodes_to_visit
                            nodes_to_visit.append((self.tree_[c], parent_prob))
            except NotFittedError as e:
                raise NotFittedError("This model is not fitted yet. Cal 'fit' \
                        with appropriate arguments before using this \
                        method.") 
            o = np.hstack(o)
            stop_time = time.time()
        else:
            o = self._predict_proba(self.estimator, X, scores)
        if self.verbose >= 1:
            print(_message_with_time("SVPClassifier", "predicting probabilities", stop_time-start_time))

        return o

    def predict_set(self, X , params):
        """ Return set-valued predictions.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Input samples.
        params : dict
            Represents parameters for the set-valued prediction task. Must contain following keys:
                - c, int
                    Representation complexity.
                - svptype, str {"fb", "dg", "sizectrl", "errorctrl"}
                    Type of set-valued predictor.
                - beta, int
                    Beta parameter in case of svptype="fb"
                - delta, float
                    Float parameter in case of svptype="dg"
                - gamma, float
                    Float parameter in case of svptype="dg"
                - size, int
                    Size parameter in case of svptype="sizectrl"
                - error, float
                    Error parameter in case of svptype="errorctrl"

        Returns
        -------
        o : list, size (n_samples,)
            Nested list of set-valued predictions.
        """
        # check input
        X = check_array(X)
        scores = False
        o = []
        start_time = time.time()
        # process params
        if type(params["c"]) != int:
            raise ValueError("Invalid representation complexity {0}. Must be integer.".format(params["c"]))
        # c must be K in case of no hierarchy
        if self.hierarchy == "none" and params["c"] < len(self.classes_):
            raise ValueError("Representation complexity {0} must be K in case of no hierarchy!".format(params["c"]))
        if params["svptype"] == "fb":
            if params["beta"] != int:
                raise ValueError("Invalid beta {0}. Must be positive integer.".format(params["beta"]))
        elif params["svptype"] == "dg":
            if params["gamma"] != float and params["delta"] != float:
                raise ValueError("Invalid delta {0} or gamma {1}. Must be positive float.".format(params["delta"], params["gamma"]))
        elif params["svptype"] == "sizectrl":
            if params["size"] != int:
                raise ValueError("Invalid size {0}. Must be positive integer.".format(params["size"]))
        elif params["svptype"] == "errorctrl":
            if params["error"] != float:
                raise ValueError("Invalid error {0}. Must be a real number in [0,1].".format(params["error"]))
        else: 
            raise ValueError("Invalid SVP type {0}! Valid options: {fb, dg, sizectrl, errorctrl}.".format(params["svptype"]))
        # check whether the base estimator supports probabilities
        if not hasattr(self.estimator, 'predict_proba'):
            # check whether the base estimator supports class scores
            if not hasattr(self.estimator, 'decision_function'):
                raise NotFittedError("{0} does not support \
                         probabilistic predictions nor scores.".format(self.estimator))
            else:
                scores = True
        try:
            # now proceed to predicting
            with parallel_backend("loky"):
                # obtain predictions
                if self.hierarchy == "none":
                    d_preds = Parallel(n_jobs=self.n_jobs)(delayed(self.__gsvbop)(i,X[ind],params,scores) for i,ind in enumerate(np.array_split(range(X.shape[0]), self.n_jobs)))
                elif params["c"] == len(self.classes_):
                    d_preds = Parallel(n_jobs=self.n_jobs)(delayed(self.__gsvbop_hf)(i,X[ind],params,scores) for i,ind in enumerate(np.array_split(range(X.shape[0]), self.n_jobs)))
                else:
                    d_preds = Parallel(n_jobs=self.n_jobs)(delayed(self.__gsvbop_hf_r)(i,X[ind],params,scores) for i,ind in enumerate(np.array_split(range(X.shape[0]), self.n_jobs)))
            # TODO: finalize
            # collect predictions
            preds_dict = dict(ChainMap(*d_preds))
            for k in np.sort(list(preds_dict.keys())):
                o.extend(preds_dict[k])
            if self.hierarchy != "none":
                # in case of no predefined hierarchy, backtransform to original labels
                if self.label_encoder_ is not None:
                    o = self.label_encoder_.inverse_transform([p.split(";")[-1] for p in o])
        except NotFittedError as e:
            raise NotFittedError("This model is not fitted yet. Cal 'fit' \
                    with appropriate arguments before using this \
                    method.")
        stop_time = time.time()
        if self.verbose >= 1:
            print(_message_with_time("SVPClassifier", "predicting set", stop_time-start_time))

        return o

    def __gsvbop(self, i, X, params, scores):
        return "Not implemented yet!"
    
    def __gsvbop_hf(self, i, X, params, scores):
        return "Not implemented yet!"
    
    def __gsvbop_hf_r(self, i, X, params, scores):
        return "Not implemented yet!"

    def score(self, X, y):
        """ Return mean accuracy score.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Test samples.
        y : array-like, shape (n_samples,)
            True labels for X.
       
        Returns
        -------
        score : float
            Mean accuracy of self.predict(X) wrt. y.
        """
        # check input and outputs
        X, y  = check_X_y(X, y, multi_output=False)
        start_time = time.time()
        try:
            preds = self.predict(X)
        except NotFittedError as e:
            raise NotFittedError("This model is not fitted yet. Cal 'fit' \
                    with appropriate arguments before using this \
                    method.")
        stop_time = time.time()
        if self.verbose >= 1:
            print(_message_with_time("SVPClassifier", "calculating score", stop_time-start_time))
        score = accuracy_score(y, preds) 
        return score

    def score_nodes(self, X, y):
        """ Return mean accuracy score for each node in the hierarchy.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Test samples.
        y : array-like, shape (n_samples,)
            True labels for X.
       
        Returns
        -------
        score_dict : dict
            Mean accuracy of self.predict(X) wrt. y for each node in the hierarchy.
        """
        # check input and outputs
        if self.hierarchy == "none":
            raise NotFittedError("Method `score_nodes` is only supported for hierarchical classifiers!")
        X, y  = check_X_y(X, y, multi_output=False)
        start_time = time.time()
        score_dict = {}
        try: 
            # transform the flat labels, in case of no predefined hierarchy
            if self.label_encoder_ is not None:
                y = self.label_encoder_.transform(y)
            for node in self.tree_:
                node = self.tree_[node]
                # check if node has estimator
                if node["estimator"] is not None:
                    # transform data for node
                    y_transform = []
                    sel_ind = []
                    for i, yi in enumerate(y):
                        if node["lbl"] in yi.split(";"):
                            # need to include current label and sample (as long as it's "complete")
                            y_split = yi.split(";")
                            if y_split.index(node["lbl"]) < len(y_split)-1:
                                y_transform.append(y_split[y_split.index(node["lbl"])+1])
                                sel_ind.append(i)
                    X_transform = X[sel_ind,:]
                    if len(sel_ind) != 0:
                        # obtain predictions
                        node_preds = node["estimator"].predict(X_transform)
                        acc = accuracy_score(y_transform, node_preds)
                        score_dict[node["lbl"]] = acc
        except NotFittedError as e:
            raise NotFittedError("This model is not fitted yet. Cal 'fit' \
                    with appropriate arguments before using this \
                    method.")
        stop_time = time.time()
        if self.verbose >= 1:
            print(_message_with_time("SVPClassifier", "calculating node scores", stop_time-start_time))
        return score_dict
