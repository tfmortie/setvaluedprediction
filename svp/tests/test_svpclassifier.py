import pickle
import numpy as np
import pandas as pd
from svp.multiclass import SVPClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split

def traintest_cal256(est):
    # first open training set
    with open("/home/data/vmoortga/data/X_train_CAL256.pickle", "rb") as f:
        X_train = pickle.load(f)
    y_train = np.array(pd.read_csv("/home/data/tfmortier/Research/Datasets/CAL256/TRAINVAL.csv", sep=",").iloc[:,1])
    X_tr, X_te, y_tr, y_te = train_test_split(X_train, y_train, test_size=0.5, random_state=2021, stratify=y_train)
    flat = SVPClassifier(est,hierarchy="none")
    hier_p = SVPClassifier(est,hierarchy="predefined")
    hier_r = SVPClassifier(est,hierarchy="random")
    print("Start fitting flat model")
    flat.fit(X_tr, y_tr)
    print("Flat model training done!")
    print("Start predefined hierarchical model")
    hier_p.fit(X_tr, y_tr)
    print("Hierarchical predefined model training done!")
    print("Start random hierarchical model")
    hier_r.fit(X_tr, y_tr)
    print("Hierarchical random model training done!")
    flat_preds = flat.predict(X_te)
    hier_p_preds = hier_p.predict(X_te)
    hier_r_preds = hier_r.predict(X_te)
    print(f'{hier_p.score_nodes(X_te, y_te)=}')
    print(f'{hier_r.score_nodes(X_te, y_te)=}')
    flat_probs = flat.predict_proba(X_te)
    hier_p_probs = hier_p.predict_proba(X_te)
    hier_r_probs = hier_r.predict_proba(X_te)
    print(f'{np.mean(flat.classes_[np.argmax(flat_probs,axis=1)]==y_te)=}')
    print(f'{np.mean(flat_preds==y_te)=}')
    print(f'{np.mean(hier_p.classes_[np.argmax(hier_p_probs,axis=1)]==y_te)=}')
    print(f'{np.mean(hier_p_preds==y_te)=}')
    print(f'{np.mean(hier_r.classes_[np.argmax(hier_r_probs,axis=1)]==y_te)=}')
    print(f'{np.mean(hier_r_preds==y_te)=}')
    params = {
        "c": 256,
        "svptype": "sizectrl",
        "size": 5
    }
    hier_r_preds = hier_r.predict(X_te)
    svp_pred = hier_r.predict_set(X_te, params)
    print(f'{svp_pred=}')
    top1 = [p[0] for p in svp_pred]
    print(f'{np.mean(top1==y_te)=}')
    print(f'{np.mean(hier_r_preds==y_te)=}')
    #print(hier_p.predict_set(X_te, params))

if __name__=="__main__":
    est = SGDClassifier(loss="log_loss") 
    traintest_cal256(est_slow)
