# Uncertainty-aware classification with set-valued predictions ![build](https://app.travis-ci.com/tfmortie/setvaluedprediction.svg?branch=main) ![pypi version](https://badge.fury.io/py/setvaluedprediction.svg) ![license](https://img.shields.io/github/license/tfmortie/setvaluedprediction)

Package for utility-based set-valued prediction in flat and hierarchical classification. 

## Description

This package provides different set-valued predictors for flat and hierarchical classification with support for Scikit-learn and PyTorch.

**TODO**: support for multi-label classification.

## Installation

Clone this repository [`tfmortie/setvaluedprediction`](https://github.com/tfmortie/setvaluedprediction.git) and run `pip install . -r requirements.txt`
or install by means of `pip install setvaluedprediction`.

## Examples 

### `SVPClassifier`

We start by importing some packages that we will need throughout the example:

```python
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
```

Creating a flat and hierarchical set-valued predictor in Scikit-learn is as simple as:

```python
from svp.multiclass import SVPClassifier

est = SGDClassifier(loss="log_loss") # classifier used for flat and hierarchical model

# create two set-valued predictors
flat = SVPClassifier(est, hierarchy="none")
hier_r = SVPClassifier(est, hierarchy="random")
```

With argument `hierarchy="random"`, we specify that no hierarchical labels are going to be provided. In this case, `SVPClassifier` automatically constructs a random hierarchy. The min and max degree of each node in the randomly generated tree can be controlled by means of the argument `k`. 

Next, we load a non-hierarchical dataset provided from Scikit-learn and split in a training and validation set:

```python
# our dataset
X, y = load_digits(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.5, random_state=2022, stratify=y)

```

`SVPClassifier` follows the Scikit-learn API, with support for standard training and inference procedures: 

```python
# training the models
flat.fit(X_tr, y_tr)
hier_r.fit(X_tr, y_tr)

# obtain predictions and class probabilities
flat_preds = flat.predict(X_te)
hier_r_preds = hier_r.predict(X_te)
flat_probs = flat.predict_proba(X_te)
hier_r_probs = hier_r.predict_proba(X_te)
```

Hence, `SVPClassifier` boils down to a standard Scikit-learn estimator but with additional support for set-valued predictions: 

```python
# initialize the set-valued predictor settings
params_flat = {
    "c": 10, # our representation complexity
    "svptype": "errorctrl", # minimize set size, while controlling the error rate
    "error": 0.01 # upper-bound the error rate by 1%
}
params_hier_r = {
    "c": 1, # our representation complexity -> in this case only internal nodes are allowed
    "svptype": "errorctrl", # minimize set size, while controlling the error rate
    "error": 0.01 # upper-bound the error rate by 1%
}

# obtain set-valued predictions
svp_preds_flat = flat.predict_set(X_te, params_flat)
svp_preds_hier_r = hier_r.predict_set(X_te, params_hier_r)
```

For more information related to the different set-valued prediction settings, see references below.

### `SVPNet`

Creating a set-valued predictor in PyTorch is very similar to `SVPClassifier`:

```python
import torch
import numpy as np
np.seterr(divide='ignore', invalid='ignore') # ignore divide by zero error's (often happens due to unstable SGDClassifier training)
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits

from svp.multiclass import SVPNet

# first load data and get training and validation sets
X, y = load_digits(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.5, random_state=2021, stratify=y)
tensor_x_tr, tensor_y_tr = torch.Tensor(X_tr), torch.Tensor(y_tr)
tensor_x_te, tensor_y_te = torch.Tensor(X_te), torch.Tensor(y_te)
dataset = TensorDataset(tensor_x_tr, tensor_y_tr) 
dataloader = DataLoader(dataset) # create your dataloader 

# create feature extractor for SGDNet and construct the set-valued predictors
phi = nn.Identity()
flat = SVPNet(phi=phi, hidden_size=X.shape[1], y, hierarchy="none")
hier_r = SVPNet(phi=phi, hidden_size=X.shape[1], classes=y, hierarchy="random")

# start fitting models
if torch.cuda.is_available():
    flat = flat.cuda()
    hier_r = hier_r.cuda()
optim_f = torch.optim.SGD(flat.parameters(), lr=0.01)
optim_hr = torch.optim.SGD(hier_r.parameters(), lr=0.01)
for _ in range(50):
    for _, data in enumerate(dataloader,1):
        inputs, labels = data
        if torch.cuda.is_available():
            inputs = inputs.cuda()
        optim_f.zero_grad()
        optim_hr.zero_grad()
        loss_f, loss_hr = flat(inputs, labels), hier_r(inputs, labels)
        loss_f.backward()
        loss_hr.backward()

# obtain top-1 predictions
if torch.cuda.is_available():
    tensor_x_te = tensor_x_te.cuda()
flat.eval()
hier_r.eval()
preds_f = flat.predict(tensor_x_test)
preds_hr = hier_r.predict(tensor_x_test)

# obtain set-valued predictions with error-rate control and maximal representation complexity
params = {
    "c": 10,
    "svptype": "sizectrl",
    "error": 0.01,
}
svp_preds_f = flat.predict_set(tensor_x_te, params)
svp_preds_hr = hier_r.predict_set(tensor_x_te, params)
```

### Set-valued prediction with predefined hierarchies

In case you want to work with predefined hierarchies, make sure that the labels are encoded in the following way:

```
# example of two hierarchical labels in some predefined hierarchy
y = ["root;Family1;Genus1;Species1", "root;Family1;Genus1;Species2"]
```

Moreover, labels are encoded as strings and correspond to paths in the predefined hierarchy where nodes are separated by `;`.

## Experiments paper(s)

* Accompanying code for paper _Set-valued prediction in hierarchical classification with constrained representation complexity_ can be found in the folder [`src/test/svphc`](./src/test/svphc).

## References

* _Efficient set-valued prediction in multi-class classification, Mortier et al., Data Mining and Knowledge Discovery (2021)_

* _Set-valued classification - overview via a unified framework, Chezhen et al., CoRR abs/2102.12318 (2021)_

* _Set-valued prediction in hierarchical classification with constrained representation complexity, Mortier et al., Proceedings of Machine Learning Research (2022)_
