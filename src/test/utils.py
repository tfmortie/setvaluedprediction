"""
Test code for utils.py.

Author: Thomas Mortier
Date: November 2021
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from main.py import utils
import numpy as np

def test_fhlabeltransformer():
    # generate a random sample with labels
    y = np.random.choice(["A", "B", "C", "D", "E", "F", "G", 
    "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", 
    "R", "S", "T", "U", "V", "W", "X", "Y", "Z"],5)
    print(f'{y=}')
    # transform labels to hierarchical labels (for some random hierarchy)
    hlt = utils.FHLabelTransformer(None,(2,4),sep=";",random_state=2021)
    y_transform = hlt.fit_transform(y)
    print(f'{y_transform=}')
    # reverse transform and check
    y_backtransform = hlt.inverse_transform(y_transform)
    print(f'{y_backtransform=}')

def test_hflabeltransformer1():
    # generate a random sample with labels
    y = np.random.choice(["A", "B", "C", "D", "E", "F", "G", 
    "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", 
    "R", "S", "T", "U", "V", "W", "X", "Y", "Z"],4)
    print(f'{y=}')
    # transform labels to hierarchical labels (for some random hierarchy)
    hlt = utils.FHLabelTransformer(None,(2,2),sep=";",random_state=2021)
    y_h = hlt.fit_transform(y)
    print(f'{y_h=}')
    # now convert to numbers in [0,K-1]
    hle = utils.HFLabelTransformer(sep=";")
    y_h_e = hle.fit_transform(y_h)
    print(f'{y_h_e=}')
    y_h_p = hle.transform(y_h, True)
    print(f'{y_h_p=}')
    # reverse transform and check
    y_h_e_backtransform = hle.inverse_transform(y_h_e)
    print(f'{y_h_e_backtransform=}')
    y_h_p_backtransform = hle.inverse_transform(y_h_p, True)
    print(f'{y_h_p_backtransform=}')
    print(f'{hle.hstruct_=}')

def test_hflabeltransformer2():
    y_h = np.array(["R;F1;G1;S1",
        "R;F1;G1;S2",
        "R;F1;G2;S3",
        "R;F1;G2;S4",
        "R;F2;G3;S5",
        "R;F2;G3;S6",
        "R;F2;G4;S7",
        "R;F2;G4;S8"])
    print(f'{y_h=}')
    # now convert to numbers in [0,K-1]
    hle = utils.HFLabelTransformer(sep=";")
    y_h_e = hle.fit_transform(y_h)
    print(f'{y_h_e=}')
    y_h_p = hle.transform(y_h, True)
    print(f'{y_h_p=}')
    # reverse transform and check
    y_h_e_backtransform = hle.inverse_transform(y_h_e)
    print(f'{y_h_e_backtransform=}')
    y_h_p_backtransform = hle.inverse_transform(y_h_p, True)
    print(f'{y_h_p_backtransform=}')
    print(f'{hle.hstruct_=}')

if __name__=="__main__":
    #print("TEST FLAT->HIERARCHICAL LABEL TRANSFORMER WITH NO STRUCT")
    #test_fhlabeltransformer()
    #print("DONE!")
    print("TEST HIERARCHICAL->FLAT LABEL TRANSFORMER 1")
    test_hflabeltransformer1()
    print("DONE!")
    print("")
    print("")
    print("TEST HIERARCHICAL->FLAT LABEL TRANSFORMER 2")
    test_hflabeltransformer2()
    print("DONE!")
