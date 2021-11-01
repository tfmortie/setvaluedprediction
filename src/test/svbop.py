import torch
from svbop_cpp import SVBOP

if __name__=="__main__":
    model = SVBOP(10, 10, [[1,2,3,4],[1,2],[3,4]])
    print("SUCCES!")