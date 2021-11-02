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
from svbop_cpp import SVBOP
from main.py.utils import HLabelEncoder, HLabelTransformer
from sklearn import preprocessing
import numpy as np

def test_svbop_flat():
    # generate a random sample with labels
    y = np.random.choice(["A", "B", "C", "D", "E", "F", "G", 
    "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", 
    "R", "S", "T", "U", "V", "W", "X", "Y", "Z"],1000)
    # now convert to numbers in [0,K-1]
    le = preprocessing.LabelEncoder()
    y = le.fit_transform(y)
    # construct PyTorch tensors
    y = torch.tensor(y)
    X = torch.randn(1000,100)
    # create softmax layer
    model = SVBOP(100,26)
    # create optimizer and criterion
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    # evaluate and backprop
    out = model(X)
    loss = criterion(out, y) 
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    print(f'{out=}')
    print(f'{loss=}')

def test_svbop_hier():
    # generate a random sample with labels
    y = np.random.choice(["A", "B", "C", "D", "E", "F", "G", 
    "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", 
    "R", "S", "T", "U", "V", "W", "X", "Y", "Z"],1000)
    hlt = HLabelTransformer(k=(2,2),sep=";",random_state=2021)
    hle = HLabelEncoder(sep=";")
    # transform labels to hierarchical labels (for some random hierarchy)
    y_t = hlt.fit_transform(y)
    # now convert to numbers in [0,K-1]
    y_t_e = hle.fit_transform(y_t) 
    # construct PyTorch tensors
    y_t_e_tensor = torch.tensor(y_t_e)
    X = torch.randn(1000,100)
    # create hierarchical softmax layer
    model = SVBOP(100,26,hle.hstruct_)
    # create optimizer and criterion
    optimizer = torch.optim.SGD(model.parameters(),lr=0.01,momentum=0.9)
    criterion = nn.BCELoss()
    # evaluate and backprop
    out = model(X,list(y_t_e))
    loss = criterion(out.view(-1),y_t_e_tensor.to(torch.float32))
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    print(f'{out=}')
    print(f'{loss=}')

if __name__=="__main__":
    print("TEST SVBOP FLAT")
    test_svbop_flat()
    print("DONE!")
    print("TEST SVBOP HIER")
    test_svbop_hier()
    print("DONE!")