""" 
Main module for experiments "Distribution-free set-valued prediction in hierarchical classification with finite sample guarantees".

Author: Thomas Mortier
Date: February 2024
"""
import time
import argparse
import torch
import pickle
import numpy as np

from svp.multiclass import SVPNet
from svp.utils import HLabelTransformer
from data import GET_DATASETLOADER
from models import GET_PHI, accuracy, recall, setsize, paramparser
from tqdm import tqdm

""" main function which trains and tests the SVP module """


def traintestsvp(args):
    # extract dataset
    dataset = args.datapath.split("/")[-1]
    print("Start reading in data for {0}...".format(dataset))
    train_data_loader, cal_data_loader, val_data_loader, classes = GET_DATASETLOADER[dataset](args)
    print("Done!")
    print("Start training and testing model...")
    # model which obtains hidden representations
    phi = GET_PHI[dataset](args)
    if args.hmodel:
        if args.randomh:
            model = SVPNet(
                phi,
                args.hidden,
                args.dropout,
                classes,
                hierarchy="random",
                random_state=args.randomseed,
            )
        else:
            model = SVPNet(
                phi,
                args.hidden,
                args.dropout,
                classes,
                hierarchy="predefined",
                random_state=args.randomseed,
            )
    else:
        model = SVPNet(
            phi, args.hidden, classes, hierarchy="none", random_state=args.randomseed
        )
    if args.gpu:
        model = model.cuda()
    # optimizer 
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.learnrate, momentum=args.momentum
    )
    #optimizer = torch.optim.Adam(model.parameters(), lr=args.learnrate)
    # train
    best_val_loss = float('inf')
    early_stop_counter = 0
    patience = 4
    for epoch in range(args.nepochs):
        train_loss, train_acc, train_time = 0.0, 0.0, 0.0
        model.train()
        for i, data in enumerate(train_data_loader, 1):
            inputs, labels = data
            labels = list(labels)
            if args.gpu:
                inputs = inputs.cuda()
            optimizer.zero_grad()
            start_time = time.time()
            loss = model(inputs, labels)
            loss.backward()
            optimizer.step()
            stop_time = time.time()
            train_time += (stop_time - start_time) / args.batchsize
            train_loss += loss.item()
            with torch.no_grad():
                preds = model.predict(inputs)
                train_acc += accuracy(preds, labels)
            #if i % args.nitprint == args.nitprint - 1:
            if i % args.nitprint == 0:
                print(
                    "Epoch {0}: training loss={1}   training accuracy={2}    training time={3}s".format(
                        epoch,
                        train_loss / args.nitprint,
                        train_acc / args.nitprint,
                        train_time / args.nitprint,
                    )
                )
                train_loss, train_acc, train_time = 0.0, 0.0, 0.0 
        # validate
        model.eval()
        val_loss = 0.0
        val_acc, val_time = 0.0, 0.0
        for i, data in enumerate(val_data_loader, 1):
            inputs, labels = data
            labels = list(labels)
            if args.gpu:
                inputs = inputs.cuda()
            with torch.no_grad():
                start_time = time.time()
                preds = model.predict(inputs)
                loss = model(inputs, labels)
                stop_time = time.time()
                val_time += (stop_time - start_time) / args.batchsize
                val_acc += accuracy(preds, labels)
                val_loss += loss.item()
        val_loss /= i
        val_acc /= i
        val_time /= i
        print("Validation loss={0}  acc={1}   test time={2}s".format(val_loss, val_acc, val_time))
        # Check for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print("Early stopping at epoch", epoch)
                break
    # validate: top-1 accuracy
    model.eval()
    val_acc, val_time = 0.0, 0.0
    for i, data in enumerate(val_data_loader, 1):
        inputs, labels = data
        labels = list(labels)
        if args.gpu:
            inputs = inputs.cuda()
        with torch.no_grad():
            start_time = time.time()
            preds = model.predict(inputs)
            stop_time = time.time()
            val_time += (stop_time - start_time) / args.batchsize
            val_acc += accuracy(preds, labels)
    print("Test accuracy={0}   test time={1}s".format(val_acc / i, val_time / i))
    for c in args.c:
        params = {"svptype": args.svptype, "c": c, "error": 0.05} # note: error is ignored during calibration
        # calibrate 
        cal_scores = []
        model.eval()
        for i, data in enumerate(tqdm(cal_data_loader), 1):
            inputs, labels = data
            labels = list(labels)
            if args.gpu:
                inputs = inputs.cuda()
            with torch.no_grad():
                start_time = time.time()
                cal_scores.extend(model.calibrate(inputs, labels, params))
                stop_time = time.time()
                val_time += (stop_time - start_time) / args.batchsize
        cal_scores = np.array(cal_scores) 
        with open("./cal_scores.pkl", "wb") as f:
            pickle.dump(cal_scores, f) 
        print("Mean NC score={0}   calibration time={1}s".format(np.mean(cal_scores), val_time / i))
        print("Number of calibration points: {}".format(len(cal_scores)))
        # validate: svp performance
        for e in args.error:
            # update error param, given NC scores
            params["error"] = e
            print(params)
            if params["svptype"] == "apsavgerrorctrl":
                idx_thresh = int(np.ceil((1-params["error"])*(1+len(cal_scores))))
                params["error"] = 1-np.sort(cal_scores)[idx_thresh]
            else:
                params["error"] = np.quantile(cal_scores, (1+(1/len(cal_scores)))*(1-params["error"]))
            print(params)
            preds_out, labels_out = [], []
            val_recall, val_setsize, val_time = [], [], 0
            probs_out = []
            for i, data in enumerate(val_data_loader, 1):
                inputs, labels = data
                labels = list(labels)
                if args.gpu:
                    inputs = inputs.cuda()
                with torch.no_grad():
                    start_time = time.time()
                    preds = model.predict_set(inputs, params)
                    if args.out != "":
                        probs = model(inputs).detach().cpu().numpy()
                    stop_time = time.time()
                    val_time += (stop_time - start_time) / args.batchsize 
                    recall_np = recall(preds, labels)
                    setsize_np = setsize(preds)
                    val_recall.append(np.mean(recall_np))
                    val_setsize.append(np.mean(setsize_np)) 
                    if args.out != "":
                        preds_out.extend(preds)
                        labels_out.extend(labels)
                        probs_out.extend(probs)
            val_recall = np.array(val_recall)
            val_setsize = np.array(val_setsize)
            if args.out != "":
                with open(
                    "./{0}_{1}_{2}.csv".format(args.out, params["c"], params["error"]), "w"
                ) as f:
                    for (pi, lj) in zip(preds_out, labels_out):
                        f.write("{0},{1}\n".format(pi, lj))
                f.close()
                probs_out = np.vstack(probs_out)
                np.save("./{0}_{1}_{2}.npy".format(args.out, params["c"], params["error"]), probs_out)
            print(
                "Test SVP for setting {0}: recall={1} +- {2}, |Ÿ|={3} +- {4}, time={5}s".format(
                    params, np.mean(val_recall), np.std(val_recall), np.mean(val_setsize), np.std(val_setsize), val_time / i
                )
            )
            print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Code for DFSVPHC experiments")
    # data args
    parser.add_argument("-cr", dest="calratio", type=float, default=0.1)
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
    parser.add_argument("-d", dest="dropout", type=float, default=0.1)
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
    parser.add_argument("-svptype", default="avgerrorctrl")
    parser.add_argument("-error", dest="error", nargs="+", type=float, default=[0.05])
    parser.add_argument("-c", dest="c", nargs="+", type=int, default=[1])
    # defaults
    parser.set_defaults(randomh=False)
    parser.set_defaults(hmodel=True)
    parser.set_defaults(gpu=True)
    args = parser.parse_args()
    # print arguments to console
    print(args)
    traintestsvp(args)