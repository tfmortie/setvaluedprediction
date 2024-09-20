"""
Models module for experiments "Distribution-free set-valued prediction in hierarchical classification with finite sample guarantees"

Author: Thomas Mortier
Date: February 2024
"""
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from itertools import product


class Identity(nn.Module):
    """Identitify layer."""

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def get_phi_bio(args):
    phi = torch.nn.Sequential(torch.nn.Linear(args.dim, args.hidden), torch.nn.ReLU())

    return phi


def get_phi_caltech(args):
    phi = models.mobilenet_v2(pretrained=True)
    phi.fx = Identity()

    return phi


def get_phi_plantclef(args):
    phi = models.mobilenet_v2(pretrained=True)
    phi.fx = Identity()

    return phi


""" dictionary representing dataset->phi mapper """
GET_PHI = {
    "CIFAR10": get_phi_caltech,
    "CAL101": get_phi_caltech,
    "CAL256": get_phi_caltech,
    "PLANTCLEF": get_phi_plantclef,
    "PROT": get_phi_bio,
    "BACT": get_phi_bio,
}

""" calculate accuracy given predictions and labels """


def accuracy(predictions, labels):
    o = np.array(predictions) == np.array(labels)

    return np.mean(o)


""" calculate recall given predictions (sets) and labels """


def recall(predictions, labels):
    recall = []
    for i, _ in enumerate(predictions):
        recall.append(int((labels[i] in predictions[i])))

    return np.array(recall)


""" calculate average set size given predictions """


def setsize(predictions):
    setsize = []
    for _, p in enumerate(predictions):
        setsize.append(len(p))

    return np.array(setsize)


""" parser for SVP parameters """


def paramparser(args):
    param_list = []
    for (c, e) in list(product(map(int, args.c), map(float, args.error))):
        params = {"svptype": args.svptype, "c": c}
        params["error"] = e
        param_list.append(params)

    return param_list
