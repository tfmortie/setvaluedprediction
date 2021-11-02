import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from main.py import utils
import numpy as np

def test_hlabelencoder():
    # generate a random sample with labels
    y = np.random.choice(["A", "B", "C", "D", "E", "F", "G", 
    "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", 
    "R", "S", "T", "U", "V", "W", "X", "Y", "Z"],5)
    hle = utils.HLabelTransformer((2,4),sep=";",random_state=2021)
    y_transform = hle.fit_transform(y)
    print(f'{y_transform=}')
    y_backtransform = hle.inverse_transform(y_transform)
    print(f'{np.all(y==y_backtransform)=}')

if __name__=="__main__":
    print("TEST HIERARCHICAL LABEL ENCODER")
    test_hlabelencoder()
    print("DONE!")