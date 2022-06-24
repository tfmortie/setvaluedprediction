"""
Test code for SVPClassifier 

Author: Thomas Mortier
Date: June 2022
"""
import pickle
import numpy as np
import pandas as pd
import numpy as np
np.seterr(divide='ignore', invalid='ignore') # ignore divide by zero error's (often happens due to unstable SGDClassifier training)
from svp.multiclass import SVPClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split

def traintest():
    est = SGDClassifier(loss="log_loss")
    # first load in training set (we will use CAL256 for)
    with open("/home/data/tfmortier/Research/Datasets/CAL256/X_train_CAL256.pickle", "rb") as f:
        X_train = pickle.load(f)
    y_train = np.array(pd.read_csv("/home/data/tfmortier/Research/Datasets/CAL256/TRAINVAL.csv", sep=",").iloc[:,1])
    X_tr, X_te, y_tr, y_te = train_test_split(X_train, y_train, test_size=0.5, random_state=2021, stratify=y_train)
    # create flat and hierarchical models
    flat = SVPClassifier(est,hierarchy="none")
    hier_p = SVPClassifier(est,hierarchy="predefined")
    hier_r = SVPClassifier(est,hierarchy="random")
    # start fitting models
    flat.fit(X_tr, y_tr)
    hier_p.fit(X_tr, y_tr)
    hier_r.fit(X_tr, y_tr)
    # obtain predictions and probabilities on validation sets
    flat_preds = flat.predict(X_te)
    hier_p_preds = hier_p.predict(X_te)
    hier_r_preds = hier_r.predict(X_te)
    flat_probs = flat.predict_proba(X_te)
    hier_p_probs = hier_p.predict_proba(X_te)
    hier_r_probs = hier_r.predict_proba(X_te)
    # check performance with score function
    print(f'{flat.score(X_te, y_te)=}')
    print(f'{hier_p.score(X_te, y_te)=}')
    print(f'{hier_r.score(X_te, y_te)=}')
    # check performance based on top-1 preds
    print(f'{np.mean(flat_preds==y_te)=}')
    print(f'{np.mean(hier_p_preds==y_te)=}')
    print(f'{np.mean(hier_r_preds==y_te)=}')
    # also check performance based on top-1 probs
    print(f'{np.mean(flat.classes_[np.argmax(flat_probs,axis=1)]==y_te)=}')
    print(f'{np.mean(hier_p.classes_[np.argmax(hier_p_probs,axis=1)]==y_te)=}')
    print(f'{np.mean(hier_r.classes_[np.argmax(hier_r_probs,axis=1)]==y_te)=}')
    # check set-valued predictions for size control with c=K
    params = {
        "c": 256,
        "svptype": "sizectrl",
        "size": 5
    }
    svp_preds_flat = flat.predict_set(X_te, params)
    svp_preds_hier_p = hier_p.predict_set(X_te, params)
    svp_preds_hier_r = hier_r.predict_set(X_te, params)
    print(f'{np.mean(np.array([len(p) for p in svp_preds_flat]))=}')
    print(f'{np.mean(np.array([len(p) for p in svp_preds_hier_p]))=}')
    print(f'{np.mean(np.array([len(p) for p in svp_preds_hier_r]))=}')

if __name__=="__main__":
    traintest()
