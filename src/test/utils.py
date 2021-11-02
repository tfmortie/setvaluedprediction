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

def test_hlabeltransformer():
    # generate a random sample with labels
    y = np.random.choice(["A", "B", "C", "D", "E", "F", "G", 
    "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", 
    "R", "S", "T", "U", "V", "W", "X", "Y", "Z"],5)
    hlt = utils.HLabelTransformer((2,4),sep=";",random_state=2021)
    y_transform = hlt.fit_transform(y)
    print(f'{y_transform=}')
    y_backtransform = hlt.inverse_transform(y_transform)
    print(f'{np.all(y==y_backtransform)=}')

def test_hlabelencoder():
    # generate a random sample with labels
    y = np.random.choice(["A", "B", "C", "D", "E", "F", "G", 
    "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", 
    "R", "S", "T", "U", "V", "W", "X", "Y", "Z"],4)
    hlt = utils.HLabelTransformer((2,2),sep=";",random_state=2021)
    y_h = hlt.fit_transform(y)
    hle = utils.HLabelEncoder(sep=";")
    y_h_e = hle.fit_transform(y_h)
    print(f'{y_h_e=}')
    y_h_e_backtransform = hle.inverse_transform(y_h_e)
    print(f'{np.all(y_h==y_h_e_backtransform)=}')
    print(f'{hle.hstruct_=}')

if __name__=="__main__":
    print("TEST HIERARCHICAL LABEL TRANSFORMER")
    test_hlabeltransformer()
    print("DONE!")
    print("TEST HIERARCHICAL LABEL ENCODER")
    test_hlabelencoder()
    print("DONE!")