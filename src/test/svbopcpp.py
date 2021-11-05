"""
Test code for svbop.cpp.

Author: Thomas Mortier
Date: November 2021
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch
import torch.nn as nn
import time
from svbop_cpp import SVBOP
from main.py.utils import HLabelEncoder, HLabelTransformer
from sklearn import preprocessing
import numpy as np

def oned2twod(y, transformer):
    y2d = []
    for yi in y:
        y2di = []
        lbls = yi.split(";") 
        for i in range(len(lbls)-2):
            par = transformer.tree_[lbls[i]]
            y2di.append(par["chn"].index(lbls[i+1]))
        y2d.append(y2di)
    return y2d

def test_svbop_flat(n, d, k):
    # generate a random sample with labels
    y = np.random.randint(0,k,n)
    # now convert to numbers in [0,K-1]
    le = preprocessing.LabelEncoder()
    y = le.fit_transform(y)
    # construct PyTorch tensors
    y = torch.tensor(y)
    X = torch.randn(n,d)
    # create softmax layer
    model = torch.nn.Sequential(
        torch.nn.Linear(d,k),
        torch.nn.Softmax(dim=1)
    )
    #model = SVBOP(d,k)
    # create optimizer and criterion
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    # evaluate and backprop
    start_time = time.time()
    out = model(X)
    stop_time = time.time()
    print("Total time forward = {0}".format(stop_time-start_time))
    start_time = time.time()
    loss = criterion(out, y) 
    loss.backward()
    optimizer.step()
    stop_time = time.time()
    print("Total time backprop = {0}".format(stop_time-start_time))
    optimizer.zero_grad()

def test_svbop_hier(n, d, k, h):
    # generate a random sample with labels
    classes = np.arange(0, k)
    y = np.random.randint(0, k, n)
    hlt = HLabelTransformer(k=(2,50),sep=";",random_state=2021)
    hle = HLabelEncoder(sep=";")
    hlt = hlt.fit(classes)
    hle = hle.fit(hlt.transform(classes))
    # transform labels to hierarchical labels (for some random hierarchy)
    y_t = hlt.transform(y)
    # now convert to numbers in [0,K-1]
    y_t_e = hle.transform(y_t) 
    y_t_l = oned2twod(y_t, hle) 
    # construct PyTorch tensors
    y_t_e_tensor = torch.tensor(y_t_e).to(torch.float32)
    X = torch.randn(n,d)
    # create hierarchical softmax layer
    model = SVBOP(d,k,hle.hstruct_)
    # create optimizer and criterion
    optimizer = torch.optim.SGD(model.parameters(),lr=0.01,momentum=0.9)
    criterion = nn.BCELoss()
    # evaluate and backprop
    start_time = time.time()
    out = model(X,y_t_l)
    stop_time = time.time()
    print("Total time forward = {0}".format(stop_time-start_time))
    start_time = time.time()
    loss = criterion(out.view(-1),y_t_e_tensor)
    loss.backward()
    optimizer.step()
    stop_time = time.time()
    print("Total time backprop = {0}".format(stop_time-start_time))
    optimizer.zero_grad()

if __name__=="__main__":
    print("TEST SVBOP HIER")
    test_svbop_hier(1, 100, 10000, (5, 10))
    print("DONE!")
    print("TEST SVBOP FLAT")
    test_svbop_flat(1, 100, 10000)
    print("DONE!")
