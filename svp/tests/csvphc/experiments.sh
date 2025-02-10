#!/bin/bash
#
# Experiments for "Conformal Prediction in Hierarchical Classification"
#
# Author: Thomas Mortier
# Date: September 2024

export LD_LIBRARY_PATH=~/.conda/envs/py310/lib:$LD_LIBRARY_PATH

# CIFAR10
python -u experiments.py -p /home/data/tfmortier/Research/Datasets/CIFAR10 --config ./experiments/ -nexp 20 -k 10 -dim 200 -hi 1000 -b 32 -ne 100 |& tee ./logs/cifar10.txt

# CAL101
python -u experiments.py -p /home/data/tfmortier/Research/Datasets/CAL101 --config ./experiments/ -np 50 -nexp 20 -k 97 -dim 200 -hi 1000 -b 32 -ne 100 |& tee ./logs/cal101.txt

# CAL256
python -u experiments.py -p /home/data/tfmortier/Research/Datasets/CAL256 --config ./experiments/ -nexp 20 -k 256 -dim 200 -hi 1000 -b 32 -ne 100 |& tee ./logs/cal256.txt

# PLANTCLEF
python -u experiments.py -p /home/data/tfmortier/Research/Datasets/PLANTCLEF --config ./experiments/ -nexp 20 -k 1000 -dim 200 -hi 1000 -b 32 -ne 100 |& tee ./logs/plantclef.txt

# BACTERIA
python -u experiments.py -p /home/data/tfmortier/Research/Datasets/BACT --config ./experiments/ -nexp 20 -k 2659 -dim 2457 -hi 1000 -b 32 -l 0.01 -ne 100 |& tee ./logs/bact.txt

# PROT
python -u experiments.py -p /home/data/tfmortier/Research/Datasets/PROT --config ./experiments/ -nexp 20 -k 3485 -dim 26276 -hi 1000 -b 32 -l 0.01 -ne 100 |& tee ./logs/prot.txt