""" 
Main module for experiments "Distribution-free set-valued prediction in hierarchical classification with finite sample guarantees".

Author: Thomas Mortier
Date: September 2024
"""
import time
import os
import argparse
import torch
import pickle
import yaml
import numpy as np
from pathlib import Path

from svp.multiclass import SVPNet
from svp.utils import HLabelTransformer
from data import GET_DATASETLOADER
from models import GET_PHI, accuracy, recall, setsize, paramparser
from tqdm import tqdm

def load_config_file(config_file):
    # Load config depending on the file format (YAML or JSON)
    if config_file.endswith('.yaml') or config_file.endswith('.yml'):
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)
    else:
        raise ValueError("Config file must be a .yaml/.yml")

def list_files_in_directory(dir):
    # Create a Path object from the directory
    directory = Path(dir)    
    # List all files directly inside the directory, excluding subdirectories
    files = [str(file) for file in directory.iterdir() if file.is_file()] 

    return files

""" main function which trains the SVP module """
def trainsvp(args):
    # extract dataset
    dataset = args.datapath.split("/")[-1]
    # make sure that we increase our seed in order to obtain a different split
    print("Extract training and validation set for {0}...".format(dataset))
    train_data_loader, val_data_loader, _, _, _, classes = GET_DATASETLOADER[dataset](args)
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
            phi, args.hidden, args.dropout, classes, hierarchy="none", random_state=args.randomseed
        )
    if args.gpu:
        model = model.cuda()
    # optimizer 
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learnrate)
    # train
    best_val_loss = float('inf')
    min_delta = 0.01
    early_stop_counter = 0
    patience = 3
    for epoch in range(args.nepochs):
        train_loss, train_time = 0.0, 0.0
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
            if i % args.nitprint == 0:
                print(
                    "Epoch {0}: training loss={1} training time={2}s".format(
                        epoch,
                        train_loss / args.nitprint,
                        train_time / args.nitprint,
                    )
                )
                train_loss, train_time = 0.0, 0.0 
                break
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
                if i % args.nitprint == 0:
                    val_loss /= args.nitprint
                    val_acc /= args.nitprint
                    val_time /= args.nitprint
                    print("Validation loss={0}  acc={1}   test time={2}s".format(val_loss, val_acc, val_time))
                    # Check for early stopping
                    if val_loss < best_val_loss - min_delta:
                        best_val_loss = val_loss
                        early_stop_counter = 0
                    else:
                        early_stop_counter += 1
                        if early_stop_counter >= patience:
                            print("Early stopping at epoch", epoch)
                    break
        if early_stop_counter >= patience:
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
    print("Done!")

    return model

""" main function which tests the SVP module """
def testsvp(args, model):
    # extract dataset
    dataset = args.datapath.split("/")[-1]
    # consider different random splits
    for n in range(args.nexp):
        # make sure that we increase our seed in order to obtain a different split
        args.randomseeddata = args.randomseeddata + n
        print(args.randomseeddata)
        print("Extract {0}-th calibration and test set for {1}...".format(n, dataset))
        _, _, cal_data_loader, tune_data_loader, test_data_loader, classes = GET_DATASETLOADER[dataset](args)
        print("Done!")
        print("Start inference...")
        if not isinstance(args.c, list):
            args.c = [args.c]
        for c in args.c:
            if c == -1:
                c = len(model.classes_)
            params = {"svptype": args.svptype, "c": c, "error": args.error, "rand": args.rand, "lambda": args.lm, "k": args.k} # note: error is ignored during calibration
            # str for outputs
            str_out = [args.out]
            if args.hmodel:
                str_out.append("h")
                if args.randomh:
                    str_out.append("r")
                else:
                    str_out.append("nr")
            else:
                str_out.append("f")
            str_out.append(str(args.error))
            str_out.append(str(params["c"]))
            str_out.append(str(n))
            if args.tunelk and args.svptype in ["raps", "csvphf", "crsvphf"]:
                # find optimal k
                probs = []
                labelidx = []
                for i, data in enumerate(tqdm(tune_data_loader), 1):
                    inputs, labels = data
                    labels = list(labels)
                    labelidx.extend([item for sublist in model.transformer.transform(labels) for item in sublist])
                    if args.gpu:
                        inputs = inputs.cuda()
                    with torch.no_grad():
                        probs.append(model(inputs).detach().cpu().numpy())
                probs_out = np.vstack(probs)
                idx_sorted = np.argsort(probs_out)[::-1]
                # now calculate the empirical quantile of the set size
                _, sizes = np.where(idx_sorted==np.array(labelidx)[:,None])
                q_level = np.ceil((len(sizes)+1)*(1-params["error"]))/len(sizes)
                # store
                params["k"] = np.quantile(sizes, q_level, method="higher")
                print("Optimal k found = {}!".format(params["k"]))
                # find optimal lambda
                lambda_list = [0, 0.001, 0.01, 0.1, 0.2, 0.5]
                opt_lambda, opt_size = 0, args.k
                for lm in lambda_list:
                    params["lambda"] = lm
                    cal_scores = []
                    for i, data in enumerate(tqdm(cal_data_loader), 1):
                        inputs, labels = data
                        labels = list(labels)
                        if args.gpu:
                            inputs = inputs.cuda()
                        with torch.no_grad():
                            cal_scores.extend(model.calibrate(inputs, labels, params))
                    cal_scores = np.array(cal_scores) 
                    # add some random noise to the scores
                    cal_scores += np.random.uniform(-1e-6, 1e-6, len(cal_scores))
                    # validate: svp performance
                    params["error"] = args.error
                    q_level = np.ceil((len(cal_scores)+1)*(1-params["error"]))/len(cal_scores)
                    params["error"] = np.quantile(cal_scores, q_level, method="higher")
                    test_setsize = []
                    for i, data in enumerate(tune_data_loader, 1):
                        inputs, labels = data
                        labels = list(labels)
                        if args.gpu:
                            inputs = inputs.cuda()
                        with torch.no_grad():
                            preds = model.predict_set(inputs, params)
                            setsize_np = setsize(preds)
                            test_setsize.append(np.mean(setsize_np)) 
                    test_setsize = np.array(test_setsize)
                    if np.mean(test_setsize) < opt_size:
                        opt_lambda, opt_size = lm, np.mean(test_setsize)
                params["lambda"] = opt_lambda
                print("Optimal lambda found = {}!".format(params["lambda"]))
            # calibrate 
            cal_scores = []
            probs = []
            val_time = 0
            model.eval()
            for i, data in enumerate(tqdm(cal_data_loader), 1):
                inputs, labels = data
                labels = list(labels)
                if args.gpu:
                    inputs = inputs.cuda()
                with torch.no_grad():
                    probs.append(model(inputs).detach().cpu().numpy())
                    start_time = time.time()
                    cal_scores.extend(model.calibrate(inputs, labels, params))
                    stop_time = time.time()
                    val_time += (stop_time - start_time) / args.batchsize
            cal_scores = np.array(cal_scores) 
            #if args.out != "":
            #    probs_out = np.vstack(probs)
            #    np.save("./{0}/{1}_calib.npy".format(dataset, "_".join(str_out)), probs_out)
            #    with open("./{0}/cal_scores_{1}.pkl".format(dataset, "_".join(str_out)), "wb") as f:
            #        pickle.dump(cal_scores, f) 
            print("Mean NC score={0}   calibration time={1}s".format(np.mean(cal_scores), val_time / i))
            print("Number of calibration points: {}".format(len(cal_scores)))
            ## add some random noise to the scores
            cal_scores += np.random.uniform(-1e-6, 1e-6, len(cal_scores))
            # validate: svp performance
            params["error"] = args.error
            print(params)
            q_level = np.ceil((len(cal_scores)+1)*(1-params["error"]))/len(cal_scores)
            params["error"] = np.quantile(cal_scores, q_level, method="higher")
            print(params)
            preds_out, labels_out = [], []
            test_recall, test_setsize, test_time = [], [], 0
            probs_out = []
            for i, data in enumerate(test_data_loader, 1):
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
                    test_time += (stop_time - start_time) / args.batchsize 
                    recall_np = recall(preds, labels)
                    setsize_np = setsize(preds)
                    test_recall.append(np.mean(recall_np))
                    test_setsize.append(np.mean(setsize_np)) 
                    if args.out != "":
                        preds_out.extend(preds)
                        labels_out.extend(labels)
                        probs_out.extend(probs)
            test_recall = np.array(test_recall)
            test_setsize = np.array(test_setsize)
            if args.out != "":
                with open("./{0}/{1}.pkl".format(dataset, "_".join(str_out)),"wb") as f: 
                    pickle.dump((preds_out,labels_out), f)
                f.close()
                probs_out = np.vstack(probs_out)
                np.save("./{0}/{1}.npy".format(dataset, "_".join(str_out)), probs_out)
            print(
                "Test SVP for setting {0}: recall={1} +- {2}, |Ÿ|={3} +- {4}, time={5}s".format(
                    params, np.mean(test_recall), np.std(test_recall), np.mean(test_setsize), np.std(test_setsize), val_time / i
                )
            )
            print("Done!")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Code for DFSVPHC experiments")
    # allow config file argument
    parser.add_argument("--config", dest="config", type=str, nargs="*")
    # data args
    parser.add_argument("-rsd", dest="randomseeddata", type=int, default=2024)
    parser.add_argument("-tr", dest="trainratio", type=float, default=0.5)
    parser.add_argument("-te", dest="testratio", type=float, default=0.5)
    parser.add_argument("-p", dest="datapath", type=str, required=True)
    parser.add_argument("-k", dest="nclasses", type=int, required=True)
    parser.add_argument("-dim", dest="dim", type=int, required=True)
    parser.add_argument("-hi", dest="hidden", type=int, required=True)
    parser.add_argument("--rh", dest="randomh", action="store_true")
    parser.add_argument("--no-rh", dest="randomh", action="store_false")
    parser.add_argument("-out", dest="out", type=str, default="")
    # model args
    parser.add_argument("-b", dest="batchsize", type=int, default=64)
    parser.add_argument("-ne", dest="nepochs", type=int, default=20)
    parser.add_argument("-d", dest="dropout", type=float, default=0.1)
    parser.add_argument("-l", dest="learnrate", type=float, default=0.0001)
    parser.add_argument("-m", dest="momentum", type=float, default=0.99)
    parser.add_argument("-np", dest="nitprint", type=int, default=100)
    parser.add_argument("-rs", dest="randomseed", type=int, default=2024)
    parser.add_argument("--hm", dest="hmodel", action="store_true")
    parser.add_argument("--no-hm", dest="hmodel", action="store_false")
    parser.add_argument("--gpu", dest="gpu", action="store_true")
    parser.add_argument("--no-gpu", dest="gpu", action="store_false")
    # SVP args
    parser.add_argument("-nexp", dest="nexp", type=int, default=10)
    parser.add_argument("-svptype", default="lac")
    parser.add_argument("-error", dest="error", type=float, default=0.1)
    parser.add_argument("--rand", dest="rand", action="store_true")
    parser.add_argument("--no-rand", dest="rand", action="store_false")
    parser.add_argument("-lm", dest="lm", type=float, default=0)
    parser.add_argument("-kreg", dest="k", type=int, default=2)
    parser.add_argument("--tunelk", dest="tunelk", action="store_true")
    parser.add_argument("--no-tunelk", dest="tunelk", action="store_false")
    parser.add_argument("-c", dest="c", nargs="+", type=int, default=[1])
    # defaults
    parser.set_defaults(randomh=False)
    parser.set_defaults(hmodel=True)
    parser.set_defaults(gpu=True)
    parser.set_defaults(rand=False)
    parser.set_defaults(tunelk=False)
    args = parser.parse_args()
    # first train probabilistic model
    model = trainsvp(args)
    if args.config:
        config_files = []
        for cfg in args.config:
            # Check if cfg is a directory
            if os.path.isdir(cfg):
                yaml_files = list_files_in_directory(cfg)
                config_files.extend(yaml_files)
            else:
                config_files.append(cfg)
        if config_files:
            print("Config file(s) have been found: {}".format(config_files))
            for cfg in config_files:
                config_data = load_config_file(cfg)
                for key, value in config_data.items():
                    if hasattr(args, key):
                        setattr(args, key, value)
                args.out = cfg.split("/")[-1].split(".")[0]
                print(args)
                testsvp(args, model)
        else:
            print("No valid config files found!")
    else:
        print("No config file(s) provided - reading from args!")
        print(args)
        testsvp(args, model)