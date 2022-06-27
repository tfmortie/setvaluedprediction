"""
Pytest test code.

Author: Thomas Mortier
Date: June 2022
"""
import torch
import numpy as np

np.seterr(
    divide="ignore", invalid="ignore"
)  # ignore divide by zero error's (often happens due to unstable SGDClassifier training)
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from svp.multiclass import SVPClassifier, SVPNet
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits

""" calculate accuracy given predictions and labels """


def accuracy(predictions, labels):
    o = np.array(predictions) == np.array(labels)

    return np.mean(o)


""" calculate recall given predictions (sets) and labels """


def recall(predictions, labels):
    recall = []
    for i, _ in enumerate(predictions):
        recall.append(int((labels[i] in predictions[i])))

    return np.mean(np.array(recall))


""" calculate average set size given predictions """


def setsize(predictions):
    setsize = []
    for _, p in enumerate(predictions):
        setsize.append(len(p))

    return np.mean(np.array(setsize))


def test_digits_sk():
    # first load data and get training and validation sets
    X, y = load_digits(return_X_y=True)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.5, random_state=2021, stratify=y
    )
    print(X_tr.shape)
    # create base estimator
    if "log_loss" not in SGDClassifier.loss_functions:
        est = SGDClassifier(loss="log")  # depecrated since v1.1
    else:
        est = SGDClassifier(loss="log_loss")
    # create models
    flat = SVPClassifier(est, hierarchy="none")
    hier_r = SVPClassifier(
        est, hierarchy="random"
    )  # we don't have a predefined hierarchy!
    # start fitting models
    flat.fit(X_tr, y_tr)
    hier_r.fit(X_tr, y_tr)
    # obtain predictions and probabilities on validation sets
    flat_preds = flat.predict(X_te)
    hier_r_preds = hier_r.predict(X_te)
    flat_probs = flat.predict_proba(X_te)
    hier_r_probs = hier_r.predict_proba(X_te)
    # check performance with score function
    print(flat.score(X_te, y_te))
    print(hier_r.score(X_te, y_te))
    # check performance based on top-1 preds
    print(np.mean(flat_preds == y_te))
    print(np.mean(hier_r_preds == y_te))
    # also check performance based on top-1 probs
    print(np.mean(flat.classes_[np.argmax(flat_probs, axis=1)] == y_te))
    print(np.mean(hier_r.classes_[np.argmax(hier_r_probs, axis=1)] == y_te))
    # check set-valued predictions for size control with c=K
    params = {"c": 256, "svptype": "sizectrl", "size": 5}
    svp_preds_flat = flat.predict_set(X_te, params)
    svp_preds_hier_r = hier_r.predict_set(X_te, params)
    print(np.mean(np.array([len(p) for p in svp_preds_flat])))
    print(np.mean(np.array([len(p) for p in svp_preds_hier_r])))
    # check set-valued predictions for error control with c=K
    params = {"c": 256, "svptype": "errorctrl", "error": 0.01}
    svp_preds_flat = flat.predict_set(X_te, params)
    svp_preds_hier_r = hier_r.predict_set(X_te, params)
    print(np.mean(np.array([len(p) for p in svp_preds_flat])))
    print(np.mean(np.array([len(p) for p in svp_preds_hier_r])))
    # check set-valued predictions for F-beta utility maximization with c=K
    params = {"c": 256, "svptype": "fb", "beta": 1}
    svp_preds_flat = flat.predict_set(X_te, params)
    svp_preds_hier_r = hier_r.predict_set(X_te, params)
    print(np.mean(np.array([len(p) for p in svp_preds_flat])))
    print(np.mean(np.array([len(p) for p in svp_preds_hier_r])))
    # check set-valued predictions for size control with c=4
    params = {"c": 4, "svptype": "sizectrl", "size": 5}
    svp_preds_hier_r = hier_r.predict_set(X_te, params)
    print(np.mean(np.array([len(p) for p in svp_preds_hier_r])))


def test_digits_nn():
    # first load data and get training and validation sets
    X, y = load_digits(return_X_y=True)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.5, random_state=2021, stratify=y
    )
    tensor_x = torch.Tensor(X_tr)
    tensor_y = torch.Tensor(y_tr)
    tensor_x_test = torch.Tensor(X_te)
    tensor_y_test = torch.Tensor(y_te)
    dataset = TensorDataset(tensor_x, tensor_y)
    dataloader = DataLoader(dataset)  # create your dataloader
    print(X_tr.shape)
    # create feature extractor
    phi = nn.Identity()
    # create models
    flat = SVPNet(phi, X.shape[1], y, hierarchy="none")
    hier_r = SVPNet(phi, X.shape[1], y, hierarchy="random")
    # start fitting models
    if torch.cuda.is_available():
        flat = flat.cuda()
        hier_r = hier_r.cuda()
    optim_f = torch.optim.SGD(flat.parameters(), lr=0.01)
    optim_hr = torch.optim.SGD(hier_r.parameters(), lr=0.01)
    for _ in range(50):
        for _, data in enumerate(dataloader, 1):
            inputs, labels = data
            if torch.cuda.is_available():
                inputs = inputs.cuda()
            optim_f.zero_grad()
            optim_hr.zero_grad()
            loss_f = flat(inputs, labels)
            loss_hr = hier_r(inputs, labels)
            loss_f.backward()
            loss_hr.backward()
    # check performance
    if torch.cuda.is_available():
        tensor_x_test = tensor_x_test.cuda()
    flat.eval()
    hier_r.eval()
    preds_f = flat.predict(tensor_x_test)
    preds_hr = hier_r.predict(tensor_x_test)
    # check set-valued predictions for size control with c=K
    params = {"c": 10, "svptype": "sizectrl", "size": 2}
    svp_preds_f = flat.predict_set(tensor_x_test, params)
    svp_preds_hr = hier_r.predict_set(tensor_x_test, params)
    print(accuracy(preds_f, tensor_y_test))
    print(accuracy(preds_hr, tensor_y_test))
    print(recall(svp_preds_f, tensor_y_test))
    print(recall(svp_preds_hr, tensor_y_test))
    print(setsize(svp_preds_f))
    print(setsize(svp_preds_hr))
    # check set-valued predictions for size control with c=2
    params = {"c": 2, "svptype": "sizectrl", "size": 2}
    svp_preds_hr = hier_r.predict_set(tensor_x_test, params)
    print(recall(svp_preds_hr, tensor_y_test))
    print(setsize(svp_preds_hr))
