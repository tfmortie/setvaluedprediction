import ast
import argparse
import pickle
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import numpy as np
import pandas as pd

from svp.utils import LabelTransformer
from data import CaltechDataset, CifarDataset, BacteriaDataset, ProteinDataset, PlantclefDataset

class RepresentationComplexity:
    def __init__(self, data):
        dataset = None
        if data in ["CAL101", "CAL256"]:
            dataset = CaltechDataset("/home/data/tfmortier/Research/Datasets/"+data+"/TRAINVAL.csv")
        elif data == "CIFAR10":
            dataset = CifarDataset("/home/data/tfmortier/Research/Datasets/"+data+"/TRAINVAL.csv")
        elif data == "PLANTCLEF":
            dataset = PlantclefDataset("/home/data/tfmortier/Research/Datasets/"+data+"/TRAINVAL.csv")
        elif data == "BACT":
            dataset = BacteriaDataset(
                dim=2457, 
                struct_path="/home/data/tfmortier/Research/Datasets/"+data+"/hierarchy_full.txt",
                svm_path="/home/data/tfmortier/Research/Datasets/"+data+"/bacteria_train_tfidf.svm",
            )
        else:
            dataset = ProteinDataset(
                struct_path="/home/data/tfmortier/Research/Datasets/"+data+"/hierarchy_full.txt",
                svm_path="/home/data/tfmortier/Research/Datasets/"+data+"/proteins_train_tfidfoh.svm",
            )
        self.lt = LabelTransformer(hierarchy="predefined")
        # fit lt
        self.lt = self.lt.fit(dataset.y)
        # get tree structure
        if data == "BACT":
            root_str = str(list(range(1,2660)))
        else:
            root_str = "root"
        self.root = str(self.lt.hlt.tree_[root_str]["yhat"])
        self.tree_transformed = {}
        for kv in self.lt.hlt.tree_:
            par = None
            if self.lt.hlt.tree_[kv]["par"] != None:
                par = str(self.lt.hlt.tree_[self.lt.hlt.tree_[kv]["par"]]["yhat"])
            chn = []
            for ch in self.lt.hlt.tree_[kv]["chn"]:
                chn.append(str(self.lt.hlt.tree_[ch]["yhat"]))
            self.tree_transformed[str(self.lt.hlt.tree_[kv]["yhat"])] = {
                "chn": chn,
                "par": par
            }

    def transform(self, Y):
        out = []
        for yhat in Y:
            if len(yhat) != 0:
                yhat_tr = [y[0] for y in self.lt.transform(yhat)]
                rcal = []
                acal = [self.root]
                while set([bi for b in rcal for bi in b]) != set(yhat_tr):
                    n = acal.pop(0)
                    if set(ast.literal_eval(n)).issubset(set(yhat_tr)):
                        rcal.append(ast.literal_eval(n))
                    elif len(set(ast.literal_eval(n)).intersection(set(yhat_tr))) != 0:
                        for ch in self.tree_transformed[n]["chn"]:
                            acal.append(ch)
                out.append(len(rcal))
        
        return out

def get_coverage_size_rep(dataset, method, model, hier, error, c, n_exps=49):
    folder = "./"+dataset+"/"
    repcf = RepresentationComplexity(data=dataset)
    coverage, size, repc = [], [], []
    for i in range(n_exps):
        # read in file
        preds, lbls = pickle.load(open(folder+method+"_"+model+"_"+hier+"_"+str(error)+"_"+str(c)+"_"+str(i)+".pkl", "rb")) 
        coverage.append(np.mean(np.array([l in p for (p,l) in zip(preds,lbls)])))
        size.append(np.mean(np.array([len(p) for p in preds])))
        repc.append(np.mean(np.array(repcf.transform(preds))))
    
    return np.array(coverage), np.array(size), np.array(repc)

def get_coverage(dataset, method, model, hier, error, c, n_exps=49):
    folder = "./"+dataset+"/"
    coverage = []
    for i in range(n_exps):
        # read in file
        preds, lbls = pickle.load(open(folder+method+"_"+model+"_"+hier+"_"+str(error)+"_"+str(c)+"_"+str(i)+".pkl", "rb")) 
        coverage.append(np.mean(np.array([l in p for (p,l) in zip(preds,lbls)])))
    
    return np.array(coverage)

def table_calculator_cov(dataset, n_exps=20):
    k = None
    if dataset == "CIFAR10":
        k = 10
    elif dataset == "CAL101":
        k = 97
    elif dataset == "CAL256":
        k = 256
    elif dataset == "PLANTCLEF":
        k = 1000
    elif dataset == "BACT":
        k = 2659
    else:
        k = 3485
    print("--- LAC")
    cov = get_coverage(dataset, "lac", "h", "nr", 0.1, k, n_exps)
    print("\\textsc{{LAC}} & {}+-{} \\\\".format(np.round(np.mean(cov),4), np.round(np.std(cov))))

    print("--- NPS")
    cov = get_coverage(dataset, "nps", "h", "nr", 0.1, k, n_exps)
    print("\\textsc{{NPS}} & {}+-{} \\\\".format(np.round(np.mean(cov),4), np.round(np.std(cov),4)))

    print("--- APS")
    cov = get_coverage(dataset, "aps", "h", "nr", 0.1, k, n_exps)
    print("\\textsc{{APS}} & {}+-{} \\\\".format(np.round(np.mean(cov),4),np.round(np.std(cov),4)))

    print("--- NPS-1")
    cov = get_coverage(dataset, "npsr", "h", "nr", 0.1, 1, n_exps)
    print("\\textsc{{NPS-1}} & {}+-{} \\\\".format(np.round(np.mean(cov),4), np.round(np.std(cov),4)))

    print("--- APS-1")
    cov = get_coverage(dataset, "apsr", "h", "nr", 0.1, 1, n_exps)
    print("\\textsc{{APS-1}} & {}+-{} \\\\".format(np.round(np.mean(cov),4), np.round(np.std(cov),4)))

    print("--- NMPS-1")
    cov = get_coverage(dataset, "nmpsr", "h", "nr", 0.1, 1, n_exps)
    print("\\textsc{{NMPS-1}} & {}+-{} \\\\".format(np.round(np.mean(cov),4), np.round(np.std(cov),4)))

    print("--- AMPS-1")
    cov = get_coverage(dataset, "ampsr", "h", "nr", 0.1, 1, n_exps)
    print("\\textsc{{AMPS-1}} & {}+-{} \\\\".format(np.round(np.mean(cov),4), np.round(np.std(cov),4)))

    print("--- NPS-3")
    cov = get_coverage(dataset, "npsr", "h", "nr", 0.1, 3, n_exps)
    print("\\textsc{{NPS-3}} & {}+-{} \\\\".format(np.round(np.mean(cov),4), np.round(np.std(cov),4)))

    print("--- APS-3")
    cov = get_coverage(dataset, "apsr", "h", "nr", 0.1, 3, n_exps)
    print("\\textsc{{APS-3}} & {}+-{} \\\\".format(np.round(np.mean(cov),4), np.round(np.std(cov),4)))
    print("------")

def table_calculator(dataset, n_exps=20):
    k = None
    if dataset == "CIFAR10":
        k = 10
    elif dataset == "CAL101":
        k = 97
    elif dataset == "CAL256":
        k = 256
    elif dataset == "PLANTCLEF":
        k = 1000
    elif dataset == "BACT":
        k = 2659
    else:
        k = 3485
    print("--- LAC")
    cov, size, rep = get_coverage_size_rep(dataset, "lac", "h", "nr", 0.1, k, n_exps)
    print("\\textsc{{LAC}} & {}+-{} & {} & {} \\\\".format(np.round(np.mean(cov),4), np.round(np.std(cov),2), np.round(np.mean(size), 4), np.round(np.mean(rep),4)))
    print("--- NPS")
    cov, size, rep = get_coverage_size_rep(dataset, "nps", "h", "nr", 0.1, k, n_exps)
    print("\\textsc{{NPS}} & {}+-{} & {} & {} \\\\".format(np.round(np.mean(cov),4), np.round(np.std(cov),2), np.round(np.mean(size), 4), np.round(np.mean(rep),4)))
    print("--- APS")
    cov, size, rep = get_coverage_size_rep(dataset, "aps", "h", "nr", 0.1, k, n_exps)
    print("\\textsc{{APS}} & {}+-{} & {} & {} \\\\".format(np.round(np.mean(cov),4),np.round(np.std(cov),2), np.round(np.mean(size), 4), np.round(np.mean(rep),4)))
    print("--- NPS-1")
    cov, size, rep = get_coverage_size_rep(dataset, "npsr", "h", "nr", 0.1, 1, n_exps)
    print("\\textsc{{NPS-1}} & {}+-{} & {} & {} \\\\".format(np.round(np.mean(cov),4), np.round(np.std(cov),2), np.round(np.mean(size), 4), np.round(np.mean(rep),4)))
    print("--- APS-1")
    cov, size, rep = get_coverage_size_rep(dataset, "apsr", "h", "nr", 0.1, 1, n_exps)
    print("\\textsc{{APS-1}} & {}+-{} & {} & {} \\\\".format(np.round(np.mean(cov),4), np.round(np.std(cov),2), np.round(np.mean(size), 4), np.round(np.mean(rep),4)))
    print("--- NMPS-1")
    cov, size, rep = get_coverage_size_rep(dataset, "nmpsr", "h", "nr", 0.1, 1, n_exps)
    print("\\textsc{{NMPS-1}} & {}+-{} & {} & {} \\\\".format(np.round(np.mean(cov),4), np.round(np.std(cov),2), np.round(np.mean(size), 4), np.round(np.mean(rep),4)))
    print("--- AMPS-1")
    cov, size, rep = get_coverage_size_rep(dataset, "ampsr", "h", "nr", 0.1, 1, n_exps)
    print("\\textsc{{AMPS-1}} & {}+-{} & {} & {} \\\\".format(np.round(np.mean(cov),4), np.round(np.std(cov),2), np.round(np.mean(size), 4), np.round(np.mean(rep),4)))
    print("--- NPS-3")
    cov, size, rep = get_coverage_size_rep(dataset, "npsr", "h", "nr", 0.1, 3, n_exps)
    print("\\textsc{{NPS-3}} & {}+-{} & {} & {} \\\\".format(np.round(np.mean(cov),4), np.round(np.std(cov),2), np.round(np.mean(size), 4), np.round(np.mean(rep),4)))
    print("--- APS-3")
    cov, size, rep = get_coverage_size_rep(dataset, "apsr", "h", "nr", 0.1, 3, n_exps)
    print("\\textsc{{APS-3}} & {}+-{} & {} & {} \\\\".format(np.round(np.mean(cov),4), np.round(np.std(cov),2), np.round(np.mean(size), 4), np.round(np.mean(rep),4)))
    print("------")

if __name__ == "__main__":
# Create an ArgumentParser object
    parser = argparse.ArgumentParser()    
    parser.add_argument(
        '--dataset', 
        type=str, 
        required=True, 
        help="Path to the dataset"
    )
    args = parser.parse_args()
    print("DATASET {}".format(args.dataset))
    table_calculator_cov(args.dataset, 20)
