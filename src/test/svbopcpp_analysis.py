"""
Softmax versus h-softmax

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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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
    forward_time = stop_time-start_time
    start_time = time.time()
    loss = criterion(out, y) 
    loss.backward()
    optimizer.step()
    stop_time = time.time()
    backprop_time = stop_time-start_time
    optimizer.zero_grad()
    return forward_time, backprop_time

def test_svbop_hier(n, d, k, h):
    # generate a random sample with labels
    classes = np.arange(0, k)
    y = np.random.randint(0, k, n)
    hlt = HLabelTransformer(k=h,sep=";",random_state=2021)
    hle = HLabelEncoder(sep=";")
    hlt = hlt.fit(classes)
    hle = hle.fit(hlt.transform(classes))
    # transform labels to hierarchical labels (for some random hierarchy)
    y_t = hlt.transform(y)
    # now convert to numbers in [0,K-1]
    y_t_e = hle.transform(y_t) 
    y_t_l = oned2twod(y_t, hle) 
    # construct PyTorch tensors
    y_t_e_tensor = torch.ones_like(torch.tensor(y_t_e)).to(torch.float32)
    #y_t_e_tensor = torch.tensor(y_t_e).to(torch.float32)
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
    forward_time = stop_time-start_time
    start_time = time.time()
    loss = criterion(out.view(-1),y_t_e_tensor)
    loss.backward()
    optimizer.step()
    stop_time = time.time()
    backprop_time = stop_time-start_time
    optimizer.zero_grad()
    return forward_time, backprop_time

if __name__=="__main__":
    fig, ax = plt.subplots(2,2,figsize=(7,7))
    # forward+backprop vs. K
    K_list = np.linspace(10,10000,20,dtype=np.int)
    t_softmax, t_hsoftmax = [],[]
    for K in K_list:
        f_time, b_time = test_svbop_flat(1, 1000, K)
        t_softmax.append(f_time+b_time)
        f_time, b_time = test_svbop_hier(1, 1000, K, (2,2))
        t_hsoftmax.append(f_time+b_time)
    ax[0,0].plot(K_list,np.array(t_softmax),"k-",K_list,np.array(t_hsoftmax),"b-")
    ax[0,0].set_title("Runtime vs K")
    # forward+backprop vs. D
    D_list = np.linspace(10,10000,20,dtype=np.int)
    t_softmax, t_hsoftmax = [],[]
    for D in D_list:
        f_time, b_time = test_svbop_flat(1, D, 10000)
        t_softmax.append(f_time+b_time)
        f_time, b_time = test_svbop_hier(1, D, 10000, (2,2))
        t_hsoftmax.append(f_time+b_time)
    ax[0,1].plot(D_list,np.array(t_softmax),"k-",D_list,np.array(t_hsoftmax),"b-")
    ax[0,1].set_title("Runtime vs D")
    # forward+backprop vs. BS
    B_list = np.linspace(1,128,20,dtype=np.int)
    t_softmax, t_hsoftmax = [],[]
    for B in B_list:
        f_time, b_time = test_svbop_flat(B, 1000, 10000)
        t_softmax.append(f_time+b_time)
        f_time, b_time = test_svbop_hier(B, 1000, 10000, (2,2))
        t_hsoftmax.append(f_time+b_time)
    ax[1,0].plot(B_list,np.array(t_softmax),"k-",B_list,np.array(t_hsoftmax),"b-")
    ax[1,0].set_title("Runtime vs BS")
    # forward+backprop vs. d
    d_list = np.linspace(2,100,20,dtype=np.int) 
    t_softmax, t_hsoftmax = [],[]
    for d in d_list:
        f_time, b_time = test_svbop_flat(1, 1000, 10000)
        t_softmax.append(f_time+b_time)
        f_time, b_time = test_svbop_hier(1, 1000, 10000, (2,d))
        t_hsoftmax.append(f_time+b_time)
    ax[1,1].plot(d_list,np.array(t_softmax),"k-",d_list,np.array(t_hsoftmax),"b-")
    ax[1,1].set_title("Runtime vs degree")
    plt.tight_layout()
    plt.savefig('./analysis.pdf',bbox_inches='tight')




