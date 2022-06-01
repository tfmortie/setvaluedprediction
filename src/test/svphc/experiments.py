"""
Main module for experiments paper "Set-valued prediction in hierarchical classification with constrained representation complexity"

Author: Thomas Mortier
Date: January 2022
"""
import sys
import time
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import torch
import torch.nn as nn
import numpy as np

from main.py.svp import SVPNet 
from data import GET_DATASETLOADER
from models import GET_PHI, accuracy, recall, setsize, paramparser
from utils import get_hstruct_tensor, pwk_ilp_get_ab, pwk_ilp

""" main function which trains and tests the SVP module """
def traintestsvp(args):
    # extract dataset
    dataset = args.datapath.split("/")[-1]
    print("Start reading in data for {0}...".format(dataset))
    train_data_loader, val_data_loader, classes = GET_DATASETLOADER[dataset](args)
    print("Done!")
    print("Start training and testing model...")
    # model which obtains hidden representations
    phi = GET_PHI[dataset](args)
    if args.hmodel:
        if args.randomh:
            model = SVPNet(phi, args.hidden, classes, sep=";", k=(2,2), random_state=args.randomseed)
        else:
            model = SVPNet(phi, args.hidden, classes, sep=";", k=None, random_state=args.randomseed)
    else:
        model = SVPNet(phi, args.hidden, classes, sep=None, k=None, random_state=args.randomseed)
    if args.gpu:
        model = model.cuda()
    # optimizer
    optimizer = torch.optim.SGD(model.parameters(),lr=args.learnrate,momentum=args.momentum)
    # train
    for epoch in range(args.nepochs):
        train_loss, train_acc, train_time = 0.0, 0.0, 0.0
        for i, data in enumerate(train_data_loader,1):
            inputs, labels = data
            labels = list(labels)
            if args.gpu:
                inputs = inputs.cuda()
            optimizer.zero_grad()
            start_time = time.time()
            loss = model(inputs, labels)
            loss.backward()
            optimizer.step()
            loss = model(inputs, labels)
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
    model.eval()
    val_acc, val_time = 0.0, 0.0
    for i, data in enumerate(val_data_loader,1):
        inputs, labels = data 
        labels = list(labels)
        if args.gpu:
            inputs = inputs.cuda()
        with torch.no_grad():
            start_time = time.time()
            preds = model.predict(inputs)
            stop_time = time.time()
            val_time += (stop_time-start_time)/args.batchsize
            val_acc += accuracy(preds, labels)
    print("Test accuracy={0}   test time={1}s".format(val_acc/i,val_time/i))
    # validate: svp performance
    params = paramparser(args)
    for param in params:
        preds_out, labels_out = [], []
        if not args.ilp:
            if not args.hmodel and param["c"]!=len(np.unique(classes)):
                hstruct = get_hstruct_tensor(model.transformer, param)
                print(f'{hstruct.shape=}')
                if args.gpu:
                    hstruct = hstruct.cuda()
                model.SVP.set_hstruct(hstruct)
            val_recall, val_setsize, val_time = 0.0, 0.0, 0.0
            for i, data in enumerate(val_data_loader,1):
                inputs, labels = data 
                labels = list(labels)
                if args.gpu:
                    inputs = inputs.cuda()
                with torch.no_grad():
                    start_time = time.time()
                    preds = model.predict_set(inputs, param)
                    stop_time = time.time()
                    val_time += (stop_time-start_time)/args.batchsize
                    val_recall += recall(preds, labels)
                    val_setsize += setsize(preds)
                    if args.out != "":
                        preds_out.extend(preds)
                        labels_out.extend(labels)
            if args.out != "":
                with open("./{0}_{1}_{2}.csv".format(args.out, param["c"], param["size"]),"w") as f:
                    for (pi,lj) in zip(preds_out, labels_out):
                        f.write("{0},{1}\n".format(pi,lj))
                f.close()
            print("Test SVP for setting {0}: recall={1}, |Ÿ|={2}, time={3}s".format(param,val_recall/i,val_setsize/i,val_time/i))
            print("Done!")
        else:
            hstruct = model.transformer.hstruct_
            A, b = pwk_ilp_get_ab(hstruct, param)
            val_recall, val_setsize, val_time = 0.0, 0.0, 0.0
            for i, data in enumerate(val_data_loader,1):
                inputs, labels = data 
                labels = list(labels)
                if args.gpu:
                    inputs = inputs.cuda()
                with torch.no_grad():
                    P = model.forward(inputs).cpu().detach().numpy()
                    preds, t = pwk_ilp(P, A, b, hstruct, args.solver, model.transformer)
                    val_time += t
                    val_recall += recall(preds, labels)
                    val_setsize += setsize(preds)

            print("Test SVP for setting {0}: recall={1}, |Ÿ|={2}, time={3}s".format(param,val_recall/i,val_setsize/i,val_time/i))

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Code for experiments of UAI paper")
    # data args
    parser.add_argument("-p", dest="datapath", type=str, required=True)
    parser.add_argument("-k", dest="nclasses", type=int, required=True)
    parser.add_argument("-dim", dest="dim", type=int, required=True)
    parser.add_argument("-hi", dest="hidden", type=int, required=True)
    parser.add_argument("--rh", dest="randomh", action="store_true")
    parser.add_argument("--no-rh", dest="randomh", action="store_false")
    parser.add_argument("-out", dest="out", type=str, default="")
    # model args
    parser.add_argument("-b", dest="batchsize", type=int, default=32)
    parser.add_argument("-ne", dest="nepochs", type=int, default=20)
    parser.add_argument("-l", dest="learnrate", type=float, default=0.0001)
    parser.add_argument("-m", dest="momentum", type=float, default=0.99)
    parser.add_argument("-np", dest="nitprint", type=int, default=100)
    parser.add_argument("-ts", dest="testsize", type=float, default=0.2)
    parser.add_argument("-rs", dest="randomseed", type=int, default=2021)
    parser.add_argument("--hm", dest="hmodel", action="store_true")
    parser.add_argument("--no-hm", dest="hmodel", action="store_false")
    parser.add_argument("--gpu", dest="gpu", action="store_true")
    parser.add_argument("--no-gpu", dest="gpu", action="store_false")
    # SVP args
    parser.add_argument("--ilp", dest="ilp", action="store_true")
    parser.add_argument("--no-ilp", dest="ilp", action="store_false")
    parser.add_argument("-solver", default="GLPK_MI", choices=["GLPK_MI", "CBC", "SCIP"])
    parser.add_argument("-constraint", default="size", choices=["none", "error", "size"])
    parser.add_argument("-beta", dest="beta", type=int, default=1)
    parser.add_argument("-error", dest="error", type=float, default=0.05)
    parser.add_argument("-size", dest="size", nargs="+", type=int, default=1)
    parser.add_argument("-c", dest="c", nargs="+", type=int, default=1)
    # defaults
    parser.set_defaults(randomh=False)
    parser.set_defaults(hmodel=True)
    parser.set_defaults(gpu=True)
    parser.set_defaults(ilp=False)
    args = parser.parse_args()
    # print arguments to console
    print(f'{args=}')
    traintestsvp(args)
