""" 
Some important classes for working with set-valued predictors.

Author: Thomas Mortier
Date: November 2021

TODO: 
    - argument checks for transformers
"""
import heapq
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import column_or_1d
from sklearn.utils.validation import check_is_fitted, check_random_state
from sklearn.exceptions import NotFittedError
from sklearn import preprocessing


class PriorityQueue:
    """HeapQ used by SVPClassifier."""

    def __init__(self):
        self.list = []

    def push(self, prob, node):
        heapq.heappush(self.list, [1 - prob, node])

    def pop(self):
        return heapq.heappop(self.list)

    def remove_all(self):
        self.list = []

    def size(self):
        return len(self.list)

    def is_empty(self):
        return len(self.list) == 0

    def __repr__(self):
        ret_str = ""
        for l in self.list:
            ret_str += "({0:.2f},{1}), ".format(1 - l[0], l[1])
        return ret_str


class LabelTransformer(TransformerMixin, BaseEstimator):
    """Label transformer for set-valued predictors.

    Parameters
    ----------

    hierarchy : {'predefined', 'random', 'none'}, default='none'
        Type of probabilistic model to consider for the label transformer.
    k : tuple of int, default=None
        Min and max number of children a node can have in the random generated tree. Is only used when hierarchy='random'.
    random_state : RandomState or an int seed, default=None
        A random number generator instance to define the state of the random permutations generator.

    Attributes
    ----------
    hierarchy : {"predefined", "random", "none"}, default="none"
        Type of probabilistic model to consider for the label transformer.
    k : tuple of int, default=None
        Min and max number of children a node can have in the random generated tree. Is only used when hierarchy='random'.
    random_state_ : RandomState or an int seed
        A random number generator instance to define the state of the
        random permutations generator.
    flt : FLabelTransformer
        Flat label transformer.
    hlt : HLabelTransformer
        Hierarchical label transformer.
    hstruct_ : list
        List BFS structure which represents the hierarchy in terms of encoded labels after fit.
    classes_ : list
        Classes (original) seen during fit.
    """

    def __init__(self, hierarchy="none", k=None, random_state=None):
        self.hierarchy = hierarchy
        if self.hierarchy not in ["predefined", "random", "none"]:
            raise ValueError(
                "Argument hierarchy must be in {'predefined', 'random', 'none'}!"
            )
        self.k = k
        self.random_state = random_state
        if self.hierarchy == "random":
            # we need to generate hierarchical labels
            if self.k is not None:
                self.flt = FLabelTransformer(
                    ";", self.k, random_state=self.random_state
                )
            else:
                self.flt = FLabelTransformer(
                    ";", (2, 2), random_state=self.random_state
                )
            self.hlt = HLabelTransformer(sep=";")
        elif self.hierarchy == "predefined":
            # hierarchical labels are provided
            self.flt = None
            self.hlt = HLabelTransformer(sep=";")
        else:
            # flat model
            self.flt = None
            self.hlt = preprocessing.LabelEncoder()

    def fit(self, y):
        """Fit label transformer.

        Parameters
        ----------
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : returns an instance of self.
        """
        # store classes seen during fit
        self.classes_ = list(np.unique(y))
        if self.hierarchy == "random":
            self.flt = self.flt.fit(y)
            self.hlt = self.hlt.fit(self.flt.transform(y))
            # store hierarchy
            self.hstruct_ = self.hlt.hstruct_
        elif self.hierarchy == "predefined":
            # check if labels are valid
            if not np.all(np.array([";" in yi for yi in y])):
                raise NotFittedError(
                    "Provided hierarchical labels are invalid. Make sure that the labels are in correct format."
                )
            self.hlt = self.hlt.fit(y)
            # store hierarchy
            self.hstruct_ = self.hlt.hstruct_
        else:
            self.hlt = self.hlt.fit(y)
            self.hstruct_ = None

        return self

    def fit_transform(self, y, path=False):
        """Fit label transformer return encoded labels.

        Parameters
        ----------
        y : array-like of shape (n_samples,)
            Target values.
        path : boolean, default=False
            Whether paths need to be returned or encoded flat labels (i.e., nodes).

        Returns
        -------
        y_transformed : array-like of shape (n_samples,)
            Represents encoded flat labels (i.e., nodes in hierarchy) or paths in hierarchy (i.e., when path is set to True) when hierarchy !="none",
            or flat labels otherwise.
        """
        self.fit(y)
        y_transformed = self.transform(y, path)

        return y_transformed

    def transform(self, y, path=False):
        """Transform labels.

        Parameters
        ----------
        y : array-like of shape (n_samples,)
            Target values.
        path : boolean, default=False
            Whether paths need to be returned or encoded flat labels (i.e., nodes).

        Returns
        -------
        y_transformed : array-like of shape (n_samples,)
            Represents encoded flat labels (i.e., nodes in hierarchy) or paths in hierarchy (i.e., when path is set to True) when hierarchy !="none", or flat labels otherwise.
        """
        if self.hierarchy == "random":
            y = self.flt.transform(y)
        if path is True:
            if self.hierarchy == "none":
                raise NotFittedError(
                    "Cannot return paths in hierarchy for flat labels."
                )
            else:
                y_transformed = self.hlt.transform(y, path)
        else:
            y_transformed = self.hlt.transform(y)

        return y_transformed

    def inverse_transform(self, y):
        """Inverse transform labels.

        Parameters
        ----------
        y : array-like of shape (n_samples,)
            Represents encoded flat labels (i.e., nodes in hierarchy) when hierarchy !="none", or flat labels otherwise.

        Returns
        -------
        y_transformed : array-like of shape (n_samples,)
        """
        if self.hierarchy != "none":
            y = [[l] for l in list(y)]
        y_transformed = self.hlt.inverse_transform(y)
        if self.hierarchy == "random":
            y_transformed = self.flt.inverse_transform(y_transformed)

        return y_transformed


class FLabelTransformer(TransformerMixin, BaseEstimator):
    """Flat to hierarchical label transformer where a hierarchy is generated by some random k-ary
    tree.

    Parameters
    ----------
    sep : str, default=';'
        String used for path encodings.
    k : tuple of int, default=(2,2)
        Min and max number of children a node can have in the random generated tree.
    random_state : RandomState or an int seed, default=None
        A random number generator instance to define the state of the
        random permutations generator.

    Attributes
    ----------
    sep : str
        String used for path encodings.
    k : tuple of int
        Represents min and max number of children a node can have in the random generated tree.
    random_state_ : RandomState or an int seed
        A random number generator instance to define the state of the
        random permutations generator.
    classes_ : list
        Classes (original) seen during fit.
    flbl_to_hlbl : Dict
        Dictionary containing key:value pairs where keys are original classes seen during fit and
        values are paths in the random generated tree.
    hlbl_to_flbl : Dict
        Reverse dictionary of flbl_to_hlbl.

    Examples
    --------
    >>> import utils
    >>> import numpy as np
    >>> y = np.random.choice(["A", "B", "C", "D", "E", "F", "G",
    "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q",
    "R", "S", "T", "U", "V", "W", "X", "Y", "Z"],1000)
    >>> hlt = utils.FLabelTransformer(sep=";", k=(2,4), random_state=2021)
    >>> y_transform = hlt.fit_transform(y)
    >>> y_backtransform = hle.inverse_transform(y_transform)
    """

    def __init__(self, sep=";", k=(2, 2), random_state=None):
        self.sep = sep
        self.k = k
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
        # store classes seen during fit
        self.classes_ = list(np.unique(y))
        # label->path in random hierarchy dict
        self.flbl_to_hlbl = {c: [] for c in self.classes_}
        # now process each unique label and get path in random hierarchy
        lbls_to_process = [[c] for c in self.classes_]
        while len(lbls_to_process) > 1:
            self.random_state_.shuffle(lbls_to_process)
            ch_list = []
            for i in range(
                min(
                    self.random_state_.randint(self.k[0], self.k[1] + 1),
                    len(lbls_to_process),
                )
            ):
                ch = lbls_to_process.pop(0)
                for c in ch:
                    self.flbl_to_hlbl[c].append(str(i))
                ch_list.extend(ch)
            lbls_to_process.append(ch_list)
        self.flbl_to_hlbl = {
            k: ".".join((v + ["r"])[::-1]) for k, v in self.flbl_to_hlbl.items()
        }
        # also store decoding dict
        self.hlbl_to_flbl = {v: k for k, v in self.flbl_to_hlbl.items()}

        return self

    def fit_transform(self, y):
        """Fit hierarchical label encoder and return encoded labels.

        Parameters
        ----------
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        y_transformed : array-like of shape (n_samples,)
        """
        self.fit(y)
        y_transformed = self.transform(y)

        return y_transformed

    def transform(self, y):
        """Transform flat labels to hierarchical encodings.

        Parameters
        ----------
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        y_transformed : array-like of shape (n_samples,)
        """
        check_is_fitted(self)
        y = column_or_1d(y, warn=True)
        y_transformed = []
        if len(y) == 0:
            # transform of empty array is empty array
            y_transformed = np.array([])
        else:
            for yi in y:
                path = self.flbl_to_hlbl[yi].split(".")
                y_transformed.append(
                    self.sep.join([".".join(path[:i]) for i in range(1, len(path) + 1)])
                )

        return y_transformed

    def inverse_transform(self, y):
        """Transform hierarchical labels back to original encodings.

        Parameters
        ----------
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        y_transformed : array-like of shape (n_samples,)
        """
        check_is_fitted(self)
        y = column_or_1d(y, warn=True)
        y_transformed = []
        if len(y) == 0:
            # transform of empty array is empty array
            y_transformed = np.array([])
        else:
            for yi in y:
                path = yi.split(self.sep)[-1]
                y_transformed.append(self.hlbl_to_flbl[path])

        return y_transformed


class HLabelTransformer(TransformerMixin, BaseEstimator):
    """Hierarchical label transformer, where hierarchical labels are encoded as nodes of hierarchy with values between 0 and n_classes-1 or paths in hierarchy.

    Parameters
    ----------
    sep : str
        String used for path encodings.

    Attributes
    ----------
    sep : str
        String used for path encodings.
    classes_ : list
        Classes (original) seen during fit.
    tree_ : Dict
        Dictionary which represents the hierarchy after fitting.
    hlbl_to_yhat_ : Dict
        Dictionary containing key:value pairs where keys are original classes seen during fit and
        values are corresponding sets of encoded labels.
    yhat_to_hlbl_ : Dict
        Reverse dictionary of hlbl_to_yhat_
    hstruct_ : list
        List BFS structure which represents the hierarchy in terms of encoded labels after fit.

    Examples
    --------
    >>> y_h = np.array(["root;famA;genA","root;famA;genB","root;famB;genC","root;famB;genD"])
    >>> hle = utils.HLabelTransformer(sep=";")
    >>> y_h_e = hle.fit_transform(y_h)
    >>> y_h_e_backtransform = hle.inverse_transform(y_h_e)
    """

    def __init__(self, sep=";"):
        self.sep = sep

    def fit(self, y):
        """Fit hierarchical label transformer.

        Parameters
        ----------
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : returns an instance of self.
        """
        y = column_or_1d(y, warn=True)
        # store classes seen during fit
        self.classes_ = list(np.unique(y))
        # now process labels and construct tree
        self.tree_ = {}
        for i, yi in enumerate(self.classes_):
            path_nodes = yi.split(self.sep)
            for j in range(1, len(path_nodes) + 1):
                node = self.sep.join(path_nodes[:j])
                if node not in self.tree_:
                    self.tree_[node] = {
                        "yhat": [i],
                        "chn": [],
                        "par": (None if j == 1 else self.sep.join(path_nodes[: j - 1])),
                    }
                    if j != 1:
                        self.tree_[self.sep.join(path_nodes[: j - 1])]["chn"].append(
                            node
                        )
                else:
                    if i not in self.tree_[node]["yhat"]:
                        self.tree_[node]["yhat"].append(i)
        # create hlbl->hpath dictionary
        self.hlbl_to_hpath_ = {c: [] for c in self.classes_}
        for c in self.classes_:
            node = c
            par_node = self.sep.join(c.split(self.sep)[:-1])
            while par_node is not None:
                self.hlbl_to_hpath_[c].append(self.tree_[par_node]["chn"].index(node))
                node = par_node
                par_node = self.tree_[node]["par"]
        # reverse values in hlbl->hpath
        self.hlbl_to_hpath_ = {k: v[::-1] for k, v in self.hlbl_to_hpath_.items()}
        # also store decoding dict
        self.hpath_to_hlbl_ = {str(v): k for k, v in self.hlbl_to_hpath_.items()}
        self.hlbl_to_yhat_ = {k: self.tree_[k]["yhat"] for k in self.tree_}
        self.yhat_to_hlbl_ = {str(v): k for k, v in self.hlbl_to_yhat_.items()}
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
            self.hstruct_.append(self.hlbl_to_yhat_[node])
            visit_list.extend([nch for nch in self.tree_[node]["chn"]])

        return self

    def fit_transform(self, y, path=False):
        """Fit hierarchical label transformer and transform hierarchical labels.

        Parameters
        ----------
        y : array-like of shape (n_samples,)
            Target values.
        path : boolean, default=False
            Whether paths need to be returned or encoded flat labels (i.e., nodes).

        Returns
        -------
        y_transformed : nested list of ints, representing encoded flat labels (nodes in hierarchy) or paths in hierarchy (when path is set to True).
        """
        self.fit(y)
        y_transformed = self.transform(y, path)

        return y_transformed

    def transform(self, y, path=False):
        """Transform hierarchical labels to encoded flat labels.

        Parameters
        ----------
        y : array-like of shape (n_samples,)
            Target values.
        path : boolean, default=False
            Whether paths need to be returned or encoded flat labels (i.e., nodes).

        Returns
        -------
        y_transformed : nested list of ints, representing encoded flat labels (i.e., nodes in hierarchy) or paths in hierarchy (i.e., when path is set to True).
        """
        check_is_fitted(self)
        y = column_or_1d(y, warn=True)
        y_transformed = []
        if len(y) == 0:
            # transform of empty array is empty array
            y_transformed = np.array([])
        else:
            for yi in y:
                if not path:
                    y_transformed.append(self.hlbl_to_yhat_[yi])
                else:
                    y_transformed.append(self.hlbl_to_hpath_[yi])

        return y_transformed

    def inverse_transform(self, y, path=False):
        """Transform encoded flat labels back to original hierarchical labels.

        Parameters
        ----------
        y : array-like of shape (n_samples,)
            Target values.
        path : boolean, default=False
            Whether y represents paths or encoded flat labels.

        Returns
        -------
        y_transformed : list of strings, representing hierarchical labels.
        """
        check_is_fitted(self)
        y_transformed = []
        if len(y) == 0:
            # transform of empty array is empty array
            y_transformed = np.array([])
        else:
            for yi in y:
                if not path:
                    y_transformed.append(self.yhat_to_hlbl_[str(yi)])
                else:
                    y_transformed.append(self.hpath_to_hlbl_[str(yi)])

        return y_transformed
