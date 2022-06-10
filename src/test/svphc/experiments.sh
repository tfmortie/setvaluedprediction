#!/bin/bash
#
# Experiments for paper "Set-valued prediction in hierarchical classification with constrained representation complexity"
#
# Author: Thomas Mortier
# Date: January 2022
#

# CAL101
python -u experiments.py -p /home/data/tfmortier/Research/Datasets/CAL101 -k 97 -dim 200 -hi 1000 -b 32 -ne 4 -size 5 10 -c 1 2 3 97 |& tee ./logs/exp_h_cal101.txt
python -u experiments.py -p /home/data/tfmortier/Research/Datasets/CAL101 -k 97 -dim 200 --no-hm -hi 1000 -b 32 -ne 2 -size 5 10 -c 1 2 3 97 |& tee ./logs/exp_f_cal101.txt
python -u experiments.py -p /home/data/tfmortier/Research/Datasets/CAL101 -k 97 -dim 200 --no-hm --ilp -hi 1000 -b 32 -ne 2 -size 5 10 -c 1 2 3 97 |& tee ./logs/exp_i_cal101.txt

# CAL256
python -u experiments.py -p /home/data/tfmortier/Research/Datasets/CAL256 -k 256 -dim 200 -hi 1000 -b 32 -ne 4 -size 5 10 -c 1 2 3 256 |& tee ./logs/exp_h_cal256.txt
python -u experiments.py -p /home/data/tfmortier/Research/Datasets/CAL256 -k 256 -dim 200 --no-hm -hi 1000 -b 32 -ne 2 -size 5 10 -c 1 2 3 256 |& tee ./logs/exp_f_cal256.txt
python -u experiments.py -p /home/data/tfmortier/Research/Datasets/CAL256 -k 256 -dim 200 --no-hm --ilp -hi 1000 -b 32 -ne 2 -size 5 10 -c 1 2 3 256 |& tee ./logs/exp_i_cal256.txt

# PlantCLEF2015
python -u experiments.py -p /home/data/tfmortier/Research/Datasets/PLANTCLEF -k 1000 -dim 200 -hi 1000 -l 0.0001 -b 32 -ne 30 -size 5 10 -c 1 2 3 1000 |& tee ./logs/exp_h_plantclef.txt
python -u experiments.py -p /home/data/tfmortier/Research/Datasets/PLANTCLEF -k 1000 -dim 200 --no-hm -hi 1000 -l 0.0001 -b 32 -ne 20 -size 5 10 -c 1 1000 |& tee ./logs/exp_f_plantclef.txt
python -u experiments.py -p /home/data/tfmortier/Research/Datasets/PLANTCLEF -k 1000 -dim 200 --no-hm --ilp -hi 1000 -l 0.0001 -b 32 -ne 20 -size 5 10 -c 1 2 3 1000 |& tee ./logs/exp_i_plantclef.txt

# BACT
python -u experiments.py -p /home/data/tfmortier/Research/Datasets/BACT -k 2659 -dim 2457 -hi 1000 -b 32 -l 0.01 -ne 30 -size 5 10 -c 1 2 3 2659 |& tee ./logs/exp_h_bact.txt
python -u experiments.py -p /home/data/tfmortier/Research/Datasets/BACT -k 2659 -dim 2457 --no-hm -hi 1000 -b 32 -l 0.01 -ne 20 -size 5 10 -c 1 2 3 2659 |& tee ./logs/exp_f_bact.txt
python -u experiments.py -p /home/data/tfmortier/Research/Datasets/BACT -k 2659 -dim 2457 --no-hm --ilp -hi 1000 -b 32 -l 0.01 -ne 20 -size 5 10 -c 1 2 3 2659 |& tee ./logs/exp_i_bact.txt

# PROT
python -u experiments.py -p /home/data/tfmortier/Research/Datasets/PROT -k 3485 -dim 26276 -hi 1000 -b 32 -l 0.01 -ne 30 -size 5 10 -c 1 2 3 3485 |& tee ./logs/exp_h_prot.txt
python -u experiments.py -p /home/data/tfmortier/Research/Datasets/PROT -k 3485 -dim 26276 --no-hm -hi 1000 -b 32 -l 0.01 -ne 20 -size 5 10 -c 1 2 3 3485 |& tee ./logs/exp_f_prot.txt
python -u experiments.py -p /home/data/tfmortier/Research/Datasets/PROT -k 3485 -dim 26276 --no-hm --ilp -hi 1000 -b 32 -l 0.01 -ne 20 -size 5 10 -c 1 2 3 3485 |& tee ./logs/exp_i_prot.txt