"""
SVP module tested on Caltech101

Author: Thomas Mortier
Date: November 2021
"""
import sys
import time
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch
import torch.nn as nn
import numpy as np
import torchvision.models as models

from svp_cpp import SVP
from main.py.utils import HFLabelTransformer, FHLabelTransformer, SVPTransformer
from main.py.svp import SVPNet 
from sklearn import preprocessing
from torchvision.datasets import Caltech101
from torchvision import transforms
from sklearn.model_selection import train_test_split

class Identity(nn.Module):
    """ Identitify layer.
    """
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

""" dataloaders for Caltech101 """
def getDataLoaders(args):
    transform = transforms.Compose([
        transforms.Lambda(lambda image: image.convert('RGB')),
        transforms.Resize((200,200)),
        transforms.ToTensor()
        ])
    dataset = Caltech101(args.datapath, download=True, transform=transform)
    train_indices, val_indices = train_test_split(list(range(len(dataset.y))), test_size=args.testsize, stratify=dataset.y)
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True, num_workers=4, drop_last=True)
    val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batchsize, shuffle=True, num_workers=4, drop_last=True)

    return train_data_loader, val_data_loader, dataset.y

""" calculate accuracy given predictions and labels """
def accuracy(predictions, labels):
    o = (np.array(predictions)==labels.cpu().detach().numpy())

    return np.mean(o)

""" calculate recall given predictions (sets) and labels """
def recall(predictions, labels):
    l = labels.cpu().detach().numpy()
    recall = []
    for i,p in enumerate(predictions):
        recall.append(int((l[i] in predictions[i])))
    
    return np.mean(np.array(recall))

""" calculate average set size given predictions """
def setsize(predictions):
    setsize = []
    for i,p in enumerate(predictions):
        setsize.append(len(p))
    
    return np.mean(np.array(setsize))

""" parser for SVP parameters """
def paramparser(args):
    params = {"constraint": args.constraint,
            "c": args.c}
    if args.constraint == "none":
        params["beta"] = args.beta
    elif args.constraint == "error":
        params["error"] = args.error
    else:
        params["size"] = args.size

    return params

""" main function which trains and tests a SVP predictor on Caltech101 """
def traintestgsvbop(args):
    print("Start reading in data...")
    # extract training and validation dataloaders
    train_data_loader, val_data_loader, classes = getDataLoaders(args)
    params = paramparser(args)
    print("Done!")
    print("Starting training and testing model...")
    # label transformer
    transformer = SVPTransformer(k=(2,2), sep=";", random_state=args.randomseed)
    transformer = transformer.fit(classes)
    # model which obtains hidden representations
    phi = models.mobilenet_v2(pretrained=True)
    phi.fx = Identity()
    # the structure representing the (random) hierarchy
    hstruct = transformer.hstruct_
    if not args.hmodel: 
        if params["c"]!=len(np.unique(classes)):
            hstruct = transformer.get_hstruct_tensor(params)
            if args.gpu:
                hstruct = hstruct.cuda()
        else:
            hstruct = None;
    # create our SVPNet model
    model = SVPNet(phi, 1000, 101, hstruct, transformer)
    if args.gpu:
        model = model.cuda()
    # optimizer
    optimizer = torch.optim.SGD(model.parameters(),lr=args.learnrate,momentum=args.momentum)
    # train
    for epoch in range(args.nepochs):
        train_loss, train_acc, train_time = 0.0, 0.0, 0.0
        for i, data in enumerate(train_data_loader,1):
            inputs, labels = data
            if args.gpu:
                inputs, labels = inputs.cuda(), labels.cuda()
            optimizer.zero_grad()
            start_time = time.time()
            loss = model(inputs, labels)
            loss.backward()
            optimizer.step()
            stop_time = time.time()
            train_time += (stop_time-start_time)/args.batchsize
            train_loss += loss.item()
            with torch.no_grad():
                preds = model.predict(inputs)
                train_acc += accuracy(preds, labels)
            if i % args.nitprint == args.nitprint-1:
                print("Epoch {0}: training loss={1}   training accuracy={2}    training time={3}s".format(epoch,train_loss/args.nitprint,
                    train_acc/args.nitprint,
                    train_time/args.nitprint))
                train_loss, train_acc, train_time = 0.0, 0.0, 0.0
    # validate: top-1 accuracy
    val_acc, val_time = 0.0, 0.0
    for i, data in enumerate(val_data_loader,1):
        inputs, labels = data 
        if args.gpu:
            inputs, labels = inputs.cuda(), labels.cuda()
        with torch.no_grad():
            start_time = time.time()
            preds = model.predict(inputs)
            stop_time = time.time()
            val_time += (stop_time-start_time)/args.batchsize
            val_acc += accuracy(preds, labels)
    print("Test accuracy={0}   test time={1}s".format(val_acc/i,val_time/i))
    # validate: svp performance
    val_recall, val_setsize, val_time = 0.0, 0.0, 0.0
    for i, data in enumerate(val_data_loader,1):
        inputs, labels = data 
        if args.gpu:
            inputs, labels = inputs.cuda(), labels.cuda()
        with torch.no_grad():
            start_time = time.time()
            preds = model.predict_set(inputs, params)
            stop_time = time.time()
            val_time += (stop_time-start_time)/args.batchsize
            val_recall += recall(preds, labels)
            val_setsize += setsize(preds)
    print("Test SVP: recall={0}, |Ÿ|={1}, time={2}s".format(val_recall/i,val_setsize/i,val_time/i))
    print("Done!")

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="SVP module tested on Caltech101 dataset")
    # data args
    parser.add_argument("-p", dest="datapath", type=str, default="/home/data/tfmortier/Research/Datasets/CAL101")
    # model args
    parser.add_argument("-b", dest="batchsize", type=int, default=32)
    parser.add_argument("-ne", dest="nepochs", type=int, default=20)
    parser.add_argument("-l", dest="learnrate", type=float, default=0.0001)
    parser.add_argument("-m", dest="momentum", type=float, default=0.99)
    parser.add_argument("-np", dest="nitprint", type=int, default=100)
    parser.add_argument("-ts", dest="testsize", type=float, default=0.2)
    parser.add_argument("-rs", dest="randomseed", type=int, default=2021)
    parser.add_argument('--hm', dest='hmodel', action='store_true')
    parser.add_argument('--no-hm', dest='hmodel', action='store_false')
    parser.add_argument('--gpu', dest='gpu', action='store_true')
    parser.add_argument('--no-gpu', dest='gpu', action='store_false')
    # SVP args
    parser.add_argument("-constraint", default='size', choices=['none', 'error', 'size'])
    parser.add_argument("-beta", dest="beta", type=int, default=1)
    parser.add_argument("-error", dest="error", type=float, default=0.05)
    parser.add_argument("-size", dest="size", type=int, default=1)
    parser.add_argument("-c", dest="c", type=int, default=1)
    args = parser.parse_args()
    # print arguments to console
    print(f'{args=}')
    traintestgsvbop(args)
