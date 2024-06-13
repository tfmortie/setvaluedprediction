#!/bin/bash
#
# Experiments for "Distribution-free set-valued prediction in hierarchical classification with finite sample guarantees"
#
# Author: Thomas Mortier
# Date: February 2024
#


# Debug
python -u experiments.py -p /home/data/tfmortier/Research/Datasets/CAL256 -k 256 -dim 200 -hi 1000 -b 32 -ne 1 -c 1 256 -error 0.05 0.10 0.15 
#python -u experiments.py -p /home/data/tfmortier/Research/Datasets/CAL256 -k 256 -dim 200 --no-hm -hi 1000 -b 32 -ne 100 -c 256 -error 0.05 0.10 0.15 
#python -u experiments.py -p /home/data/tfmortier/Research/Datasets/CAL256 --rh -k 256 -dim 200 -hi 1000 -b 32 -ne 100 -c 1 256 -error 0.05 0.10 0.15 

# CAL101 
#python -u experiments.py -p /home/data/tfmortier/Research/Datasets/CAL101 -k 97 -dim 200 --no-hm -hi 1000 -b 32 -ne 2 -c 97 -error 0.05 0.10 0.15 
#python -u experiments.py -p /home/data/tfmortier/Research/Datasets/CAL101 -k 97 -dim 200 -hi 1000 -b 32 -ne 100 -c 1 97 -error 0.05 0.10 0.15 |& tee ./logs/exp_h_cal101.txt
#python -u experiments.py -p /home/data/tfmortier/Research/Datasets/CAL101 -k 97 -dim 200 --no-hm -hi 1000 -b 32 -ne 100 -c 97 -error 0.05 0.10 0.15 |& tee ./logs/exp_f_cal101.txt
#python -u experiments.py -p /home/data/tfmortier/Research/Datasets/CAL101 --rh -k 97 -dim 200 -hi 1000 -b 32 -ne 100 -c 1 97 -error 0.05 0.10 0.15 |& tee ./logs/exp_rh_cal101.txt
#
## CAL256
#python -u experiments.py -p /home/data/tfmortier/Research/Datasets/CAL256 -k 256 -dim 200 --no-hm -hi 1000 -b 32 -ne 4 -c 1 256 -error 0.05 0.10 0.15 
#python -u experiments.py -p /home/data/tfmortier/Research/Datasets/CAL256 -k 256 -dim 200 -hi 1000 -b 32 -ne 100 -c 1 256 -error 0.05 0.10 0.15 |& tee ./logs/exp_h_cal256.txt
#python -u experiments.py -p /home/data/tfmortier/Research/Datasets/CAL256 -k 256 -dim 200 --no-hm -hi 1000 -b 32 -ne 100 -c 256 -error 0.05 0.10 0.15 |& tee ./logs/exp_f_cal256.txt
#python -u experiments.py -p /home/data/tfmortier/Research/Datasets/CAL256 --rh -k 256 -dim 200 -hi 1000 -b 32 -ne 100 -c 1 256 -error 0.05 0.10 0.15 |& tee ./logs/exp_rh_cal256.txt
#
## PlantCLEF2015
#python -u experiments.py -p /home/data/tfmortier/Research/Datasets/PLANTCLEF -k 1000 -dim 200 -hi 1000 -l 0.0001 -b 32 -ne 100 -c 1 1000 -error 0.05 0.10 0.15 |& tee ./logs/exp_h_plantclef.txt
#python -u experiments.py -p /home/data/tfmortier/Research/Datasets/PLANTCLEF -k 1000 -dim 200 --no-hm -hi 1000 -l 0.0001 -b 32 -ne 100 -c 1000 -error 0.05 0.10 0.15 |& tee ./logs/exp_f_plantclef.txt
#python -u experiments.py -p /home/data/tfmortier/Research/Datasets/PLANTCLEF --rh -k 1000 -dim 200 -hi 1000 -l 0.0001 -b 32 -ne 100 -c 1 1000 -error 0.05 0.10 0.15 |& tee ./logs/exp_rh_plantclef.txt
#
## BACT
#python -u experiments.py -p /home/data/tfmortier/Research/Datasets/BACT -k 2659 -dim 2457 -hi 1000 -l 0.01 -b 32 -ne 100 -c 1 2659 -error 0.05 0.10 0.15 |& tee ./logs/exp_h_bact.txt
#python -u experiments.py -p /home/data/tfmortier/Research/Datasets/BACT -k 2659 -dim 2457 --no-hm -hi 1000 -l 0.01 -b 32 -ne 100 -c 2659 -error 0.05 0.10 0.15 |& tee ./logs/exp_f_bact.txt
#python -u experiments.py -p /home/data/tfmortier/Research/Datasets/BACT --rh -k 2659 -dim 2457 -hi 1000 -l 0.01 -b 32 -ne 100 -c 1 2659 -error 0.05 0.10 0.15 |& tee ./logs/exp_rh_bact.txt
#
# PROT
#python -u experiments.py -p /home/data/tfmortier/Research/Datasets/PROT -k 3485 -dim 26276 -hi 1000 -l 0.01 -b 32 -ne 100 -c 1 3485 -error 0.05 0.10 0.15 |& tee ./logs/exp_h_prot.txt
#python -u experiments.py -p /home/data/tfmortier/Research/Datasets/PROT -k 3485 -dim 26276 --no-hm -hi 1000 -l 0.01 -b 32 -ne 100 -c 3485 -error 0.05 0.10 0.15 |& tee ./logs/exp_f_prot.txt
#python -u experiments.py -p /home/data/tfmortier/Research/Datasets/PROT --rh -k 3485 -dim 26276 -hi 1000 -l 0.01 -b 32 -ne 100 -c 1 3485 -error 0.05 0.10 0.15 |& tee ./logs/exp_rh_prot.txt

# OLD EXPERIMENTS

# CAL101
#python -u experiments.py -p /home/data/tfmortier/Research/Datasets/CAL101 -k 97 -dim 200 -hi 1000 -b 32 -ne 2 -c 1 97 |& tee ./logs/exp_h_cal101.txt
#python -u experiments.py -p /home/data/tfmortier/Research/Datasets/CAL101 --rh -k 97 -dim 200 -hi 1000 -b 32 -ne 2 -c 1 97 |& tee ./logs/exp_rh_cal101.txt
#python -u experiments.py -p /home/data/tfmortier/Research/Datasets/CAL101 -k 97 -dim 200 -hi 1000 -b 32 -ne 4 -c 1 97 |& tee ./logs/exp_h_cal101.txt
#python -u experiments.py -p /home/data/tfmortier/Research/Datasets/CAL101 -k 97 -dim 200 --no-hm -hi 1000 -b 32 -ne 100 -c 97 |& tee ./logs/exp_f_cal101.txt

# CAL256
#python -u experiments.py -p /home/data/tfmortier/Research/Datasets/CAL256 -k 256 -dim 200 -hi 1000 -b 32 -ne 4 -size 5 10 -c 1 2 3 256 |& tee ./logs/exp_h_cal256.txt
#python -u experiments.py -p /home/data/tfmortier/Research/Datasets/CAL256 -k 256 -dim 200 --no-hm -hi 1000 -b 32 -ne 2 -size 5 10 -c 1 2 3 256 |& tee ./logs/exp_f_cal256.txt
#python -u experiments.py -p /home/data/tfmortier/Research/Datasets/CAL256 -k 256 -dim 200 --no-hm --ilp -hi 1000 -b 32 -ne 2 -size 5 10 -c 1 2 3 256 |& tee ./logs/exp_i_cal256.txt
#
## PlantCLEF2015
#python -u experiments.py -p /home/data/tfmortier/Research/Datasets/PLANTCLEF -k 1000 -dim 200 -hi 1000 -l 0.0001 -b 32 -ne 30 -size 5 10 -c 1 2 3 1000 |& tee ./logs/exp_h_plantclef.txt
#python -u experiments.py -p /home/data/tfmortier/Research/Datasets/PLANTCLEF -k 1000 -dim 200 --no-hm -hi 1000 -l 0.0001 -b 32 -ne 20 -size 5 10 -c 1 1000 |& tee ./logs/exp_f_plantclef.txt
#python -u experiments.py -p /home/data/tfmortier/Research/Datasets/PLANTCLEF -k 1000 -dim 200 --no-hm --ilp -hi 1000 -l 0.0001 -b 32 -ne 20 -size 5 10 -c 1 2 3 1000 |& tee ./logs/exp_i_plantclef.txt
#
## BACT
#python -u experiments.py -p /home/data/tfmortier/Research/Datasets/BACT -k 2659 -dim 2457 -hi 1000 -b 32 -l 0.01 -ne 30 -size 5 10 -c 1 2 3 2659 |& tee ./logs/exp_h_bact.txt
#python -u experiments.py -p /home/data/tfmortier/Research/Datasets/BACT -k 2659 -dim 2457 --no-hm -hi 1000 -b 32 -l 0.01 -ne 20 -size 5 10 -c 1 2 3 2659 |& tee ./logs/exp_f_bact.txt
#python -u experiments.py -p /home/data/tfmortier/Research/Datasets/BACT -k 2659 -dim 2457 --no-hm --ilp -hi 1000 -b 32 -l 0.01 -ne 20 -size 5 10 -c 1 2 3 2659 |& tee ./logs/exp_i_bact.txt
#
## PROT
#python -u experiments.py -p /home/data/tfmortier/Research/Datasets/PROT -k 3485 -dim 26276 -hi 1000 -b 32 -l 0.01 -ne 30 -size 5 10 -c 1 2 3 3485 |& tee ./logs/exp_h_prot.txt
#python -u experiments.py -p /home/data/tfmortier/Research/Datasets/PROT -k 3485 -dim 26276 --no-hm -hi 1000 -b 32 -l 0.01 -ne 20 -size 5 10 -c 1 2 3 3485 |& tee ./logs/exp_f_prot.txt
#python -u experiments.py -p /home/data/tfmortier/Research/Datasets/PROT -k 3485 -dim 26276 --no-hm --ilp -hi 1000 -b 32 -l 0.01 -ne 20 -size 5 10 -c 1 2 3 3485 |& tee ./logs/exp_i_prot.txt
