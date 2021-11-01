import torch
from svbop_cpp import HSoftmax

if __name__=="__main__":
    model = HSoftmax(10,10)

    print("SUCCES!")