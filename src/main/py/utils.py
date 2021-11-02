"""
Some important functions and classes that are used throughout the module.

Author: Thomas Mortier
Date: November 2021
"""
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import column_or_1d
from sklearn.utils.validation import check_is_fitted, check_random_state

class HLabelTransformer(TransformerMixin, BaseEstimator):
    """ Hierarchical label transformer which tranforms flat labels to hierarchical labels
    for some random generated hierarchy, represented by a k-ary tree.

    Parameters
    ----------
    k : tuple of int, default=(2,2)
        Min and max number of children a node can have in the random generated tree. Is ignored when
        sep is set to None.
    sep : str, default=';'
        String used for path encodings.
    random_state : RandomState or an int seed, default=None
        A random number generator instance to define the state of the
        random permutations generator.
    
    Attributes
    ----------
    k : tuple of int
        Represents min and max number of children a node can have in the random generated tree.
    sep : str
        String used for path encodings.
    random_state_ : RandomState or an int seed
        A random number generator instance to define the state of the
        random permutations generator.
    classes_ : list 
        Classes (original) seen during fit.
    flbl_to_tlbl : Dict
        Dictionary containing key:value pairs where keys are original classes seen during fit and 
        values are paths in the random generated tree.
    tlbl_to_flbl : Dict
        Reverse dictionary of flbl_to_tlbl.
    
    Examples
    --------
    >>> import utils
    >>> import numpy as np
    >>> y = np.random.choice(["A", "B", "C", "D", "E", "F", "G", 
    "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", 
    "R", "S", "T", "U", "V", "W", "X", "Y", "Z"],1000)
    >>> hlt = utils.HLabelTransformer((2,4),sep=";",random_state=2021)
    >>> y_transform = hlt.fit_transform(y)
    >>> y_backtransform = hle.inverse_transform(y_transform)
    """
    def __init__(self, k=(2,2), sep=';', random_state=None):
        self.k = k
        self.sep = sep
        self.random_state = random_state

    def fit(self, y):
        """Fit hierarchical label encoder.
        
        Parameters
        ----------
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : returns an instance of self.
        """
        self.random_state_ = check_random_state(self.random_state)
        y = column_or_1d(y, warn=True)
        if type(self.sep) != str:
            raise TypeError("Parameter sep must be of type str.")
        # store classes seen during fit
        self.classes_ = list(np.unique(y))
        # label->path in random hierarchy dict
        self.flbl_to_tlbl = {c:[] for c in self.classes_}
        # now process each unique label and get path in random hierarchy
        lbls_to_process = [[c] for c in self.classes_]
        while len(lbls_to_process) > 1:
            self.random_state_.shuffle(lbls_to_process)
            ch_list = []
            for i in range(min(self.random_state_.randint(self.k[0], self.k[1]+1),len(lbls_to_process))):
                ch = lbls_to_process.pop(0)
                for c in ch:
                    self.flbl_to_tlbl[c].append(str(i))
                ch_list.extend(ch)
            lbls_to_process.append(ch_list)
        self.flbl_to_tlbl = {k: '.'.join((v+['r'])[::-1]) for k,v in self.flbl_to_tlbl.items()}
        # also store decoding dict
        self.tlbl_to_flbl = {v:k for k,v in self.flbl_to_tlbl.items()}

        return self

    def fit_transform(self, y):
        """Fit hierarchical label encoder and return encoded labels.

        Parameters
        ----------
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        y : array-like of shape (n_samples,)
        """
        self.fit(y)

        return self.transform(y)

    def transform(self, y):
        """Transform labels to normalized encoding.

        Parameters
        ----------
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        y : array-like of shape (n_samples,)
        """
        check_is_fitted(self)
        y = column_or_1d(y, warn=True)
        # transform of empty array is empty array
        if len(y) == 0:
            return np.array([])
        y_transformed = []
        for yi in y:
            path = self.flbl_to_tlbl[yi].split('.')
            y_transformed.append(self.sep.join(['.'.join(path[:i]) for i in range(1,len(path)+1)]))

        return y_transformed

    def inverse_transform(self, y):
        """Transform labels back to original encoding.

        Parameters
        ----------
        y : ndarray of shape (n_samples,)
            Target values.

        Returns
        -------
        y : ndarray of shape (n_samples,)
        """
        check_is_fitted(self)
        y = column_or_1d(y, warn=True)
        # transform of empty array is empty array
        if len(y) == 0:
            return np.array([])
        y_transformed = []
        for yi in y:
            path = yi.split(self.sep)[-1]
            y_transformed.append(self.tlbl_to_flbl[path])

        return y_transformed

class HLabelEncoder(TransformerMixin, BaseEstimator):
    """ Hierarchical label encoder which encodes hierarchical target labels to values between 0 and n_classes-1.     

    Parameters
    ----------
    sep : str
        String used for path encodings.
    classes_ : list 
        Classes (original) seen during fit.
    tree_ : Dict
        Dictionary which represents the taxonomy after fitting.
    lbl_to_yhat_ : Dict
        Dictionary containing key:value pairs where keys are original classes seen during fit and 
        values are corresponding sets of encoded labels.
    hstruct_ : list
        List BFS structure which represents the taxonomy in terms of encoded labels after fit.
    
    Attributes
    ----------
    sep : str
        String used for path encodings.
     
    Examples
    --------
    >>> y_h = np.array(["root;famA;genA","root;famA;genB","root;famB;genC","root;famB;genD"])
    >>> hle = utils.HLabelEncoder(sep=";")
    >>> y_h_e = hle.fit_transform(y_h)
    >>> y_h_e_backtransform = hle.inverse_transform(y_h_e)
    """
    def __init__(self, sep=";"):
        self.sep = sep
    
    def fit(self, y):
        """Fit hierarchical label encoder.
        
        Parameters
        ----------
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : returns an instance of self.
        """
        y = column_or_1d(y, warn=True)
        if type(self.sep) != str:
            raise TypeError("Parameter sep must be of type str.")
        # store classes seen during fit
        self.classes_ = list(np.unique(y))
        # now process labels and construct tree
        self.tree_ = {}
        for i,yi in enumerate(self.classes_):
            path_nodes = yi.split(self.sep)
            for j,nj in enumerate(path_nodes):
                if nj not in self.tree_:
                    self.tree_[nj] = {
                        "yhat": [i],
                        "chn": [],
                        "par": (None if j==0 else path_nodes[j-1]),
                        "lbl": (nj if j==0 else self.tree_[path_nodes[j-1]]["lbl"]+self.sep+nj)
                    }
                    if j!=0:
                        self.tree_[path_nodes[j-1]]["chn"].append(nj)
                else:
                    if i not in self.tree_[nj]["yhat"]:
                        self.tree_[nj]["yhat"].append(i)
        # now obtain lbl->yhat mapping
        self.lbl_to_yhat_ = {self.tree_[k]["lbl"]:self.tree_[k]["yhat"] for k,v in self.tree_.items()}
        print(f'{self.lbl_to_yhat_=}')
        # and obtain struct (in terms of yhat)
        self.hstruct_ = []
        # find the root first 
        root = None
        for n in self.tree_:
            if self.tree_[n]["par"] is None:
                root = n
                break
        visit_list = [root]
        # now start constructing the struct
        while len(visit_list) != 0:
            node = visit_list.pop(0)
            self.hstruct_.append(self.lbl_to_yhat_[node])
            visit_list.extend([node+self.sep+nch for nch in self.tree_[node.split(self.sep)[-1]]["chn"]])

        return self

    def fit_transform(self, y):
        """Fit hierarchical label encoder and return encoded labels.

        Parameters
        ----------
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        y : array-like of shape (n_samples,)
        """
        self.fit(y)

        return self.transform(y)

    def transform(self, y):
        """Transform labels to normalized encoding.

        Parameters
        ----------
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        y : array-like of shape (n_samples,)
        """
        check_is_fitted(self)
        y = column_or_1d(y, warn=True)
        # transform of empty array is empty array
        if len(y) == 0:
            return np.array([])
        y_transformed = []
        for yi in y:
            y_transformed.append(self.lbl_to_yhat_[yi][0])

        return y_transformed

    def inverse_transform(self, y):
        """Transform labels back to original encoding.

        Parameters
        ----------
        y : ndarray of shape (n_samples,)
            Target values.

        Returns
        -------
        y : ndarray of shape (n_samples,)
        """
        check_is_fitted(self)
        y = column_or_1d(y, warn=True)
        # transform of empty array is empty array
        if len(y) == 0:
            return np.array([])
        # create reverse yhat->lbl mappint
        yhat_to_lbl = {str(v):k for (k,v) in self.lbl_to_yhat_.items()}
        y_transformed = []
        for yi in y:
            y_transformed.append(yhat_to_lbl[str([yi])])

        return y_transformed