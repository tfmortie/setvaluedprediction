#!/bin/bash

python -u tablecalculator.py --dataset CIFAR10  |& tee ./logs/table_cifar10.txt
python -u tablecalculator.py --dataset CAL101  |& tee ./logs/table_cal101.txt
python -u tablecalculator.py --dataset CAL256  |& tee ./logs/table_cal256.txt
python -u tablecalculator.py --dataset PLANTCLEF  |& tee ./logs/table_plantclef.txt