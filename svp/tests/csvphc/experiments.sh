#!/bin/bash
#
# Experiments for "Distribution-free set-valued prediction in hierarchical classification with finite sample guarantees"
#
# Author: Thomas Mortier
# Date: February 2024
#


# New experiments 19/09/2024
# Debug
#python -u experiments.py -p /home/data/tfmortier/Research/Datasets/CIFAR10 -out CIFAR10 -nexp 2 -k 10 -dim 200 -hi 1000 -b 32 -ne 100 -c 1 10 -error 0.10 -svptype avgerrorctrl |& tee ./logs/exp_h_noraps_cifar10.txt
python -u experiments.py -p /home/data/tfmortier/Research/Datasets/CIFAR10 -out CIFAR10DEBUG -nexp 2 -k 10 -dim 200 -hi 1000 -b 64 -ne 1 -c 10 -error 0.10 -svptype avgerrorctrl --no-hm |& tee ./logs/exp_f_noraps_cifar10.txt

# Debug
#python -u experiments.py -p /home/data/tfmortier/Research/Datasets/CAL256 -k 256 -dim 200 -hi 1000 -b 32 -ne 100 -c 1 256 -error 0.05 0.10 0.15 -svptype rapsavgerrorctrl -lm 0.0 --rand |& tee ./logs/exp_h_raps_random_l0_cal256.txt
#python -u experiments.py -p /home/data/tfmortier/Research/Datasets/CAL256 -k 256 -dim 200 -hi 1000 -b 32 -ne 100 -c 256 -error 0.05 0.10 0.15 -svptype rapsavgerrorctrl -lm 0.0 --rand --no-hm |& tee ./logs/exp_f_raps_random_l0_cal256.txt # ERROR 

#python -u experiments.py -p /home/data/tfmortier/Research/Datasets/CAL256 -k 256 -dim 200 -hi 1000 -b 32 -ne 100 -c 1 256 -error 0.05 0.10 0.15 -svptype rapsavgerrorctrl -lm 0.0 --no-rand  |& tee ./logs/exp_h_raps_nonrandom_l0_cal256.txt
#python -u experiments.py -p /home/data/tfmortier/Research/Datasets/CAL256 -k 256 -dim 200 -hi 1000 -b 32 -ne 100 -c 256 -error 0.05 0.10 0.15 -svptype rapsavgerrorctrl -lm 0.0 --no-rand  --no-hm |& tee ./logs/exp_f_raps_nonrandom_l0_cal256.txt # ERROR
#
#python -u experiments.py -p /home/data/tfmortier/Research/Datasets/CAL256 -k 256 -dim 200 -hi 1000 -b 32 -ne 100 -c 1 256 -error 0.05 0.10 0.15 -svptype rapsavgerrorctrl -lm 0.5 -kreg 5 --rand |& tee ./logs/exp_h_raps_random_l05_k5_cal256.txt
#python -u experiments.py -p /home/data/tfmortier/Research/Datasets/CAL256 -k 256 -dim 200 -hi 1000 -b 32 -ne 100 -c 256 -error 0.05 0.10 0.15 -svptype rapsavgerrorctrl -lm 0.5 -kreg 5 --rand --no-hm |& tee ./logs/exp_f_raps_random_l05_k5_cal256.txt # ERROR
#
#python -u experiments.py -p /home/data/tfmortier/Research/Datasets/CAL256 -k 256 -dim 200 -hi 1000 -b 32 -ne 100 -c 1 256 -error 0.05 0.10 0.15 -svptype avgerrorctrl |& tee ./logs/exp_h_avgerror_cal256.txt 
#python -u experiments.py -p /home/data/tfmortier/Research/Datasets/CAL256 -k 256 -dim 200 -hi 1000 -b 32 -ne 100 -c 256 -error 0.05 0.10 0.15 -svptype avgerrorctrl --no-hm |& tee ./logs/exp_f_avgerror_cal256.txt # ERROR

#
#python -u experiments.py -p /home/data/tfmortier/Research/Datasets/CAL256 -k 256 -dim 200 -hi 1000 -b 32 -ne 2 -c 1 256 -error 0.05 0.10 0.15 -svptype avgerrorctrl |& tee ./logs/exp_h_noraps_cal256.txt
#python -u experiments.py -p /home/data/tfmortier/Research/Datasets/CAL256 -k 256 -dim 200 -hi 1000 -b 32 -ne 2 -c 1 256 -error 0.05 0.10 0.15 -svptype avgerrorctrl |& tee ./logs/exp_h_noraps_cal256.txt


# New experiments 04/08/2024

# CAL256
#python -u experiments.py -p /home/data/tfmortier/Research/Datasets/CAL256 -k 256 -dim 200 -hi 1000 -b 32 -ne 100 -c 1 256 -error 0.05 0.10 0.15 -svptype rapsavgerrorctrl -lm 0.0 --rand |& tee ./logs/exp_h_raps_random_l0_cal256.txt
#python -u experiments.py -p /home/data/tfmortier/Research/Datasets/CAL256 -k 256 -dim 200 -hi 1000 -b 32 -ne 100 -c 256 -error 0.05 0.10 0.15 -svptype rapsavgerrorctrl -lm 0.0 --rand --no-hm |& tee ./logs/exp_f_raps_random_l0_cal256.txt
#
#python -u experiments.py -p /home/data/tfmortier/Research/Datasets/CAL256 -k 256 -dim 200 -hi 1000 -b 32 -ne 100 -c 1 256 -error 0.05 0.10 0.15 -svptype rapsavgerrorctrl -lm 0.0 --no-rand  |& tee ./logs/exp_h_raps_nonrandom_l0_cal256.txt
#python -u experiments.py -p /home/data/tfmortier/Research/Datasets/CAL256 -k 256 -dim 200 -hi 1000 -b 32 -ne 100 -c 256 -error 0.05 0.10 0.15 -svptype rapsavgerrorctrl -lm 0.0 --no-rand  --no-hm |& tee ./logs/exp_f_raps_nonrandom_l0_cal256.txt 
#
#python -u experiments.py -p /home/data/tfmortier/Research/Datasets/CAL256 -k 256 -dim 200 -hi 1000 -b 32 -ne 100 -c 1 256 -error 0.05 0.10 0.15 -svptype rapsavgerrorctrl -lm 0.5 -kreg 5 --rand |& tee ./logs/exp_h_raps_random_l05_k5_cal256.txt
#python -u experiments.py -p /home/data/tfmortier/Research/Datasets/CAL256 -k 256 -dim 200 -hi 1000 -b 32 -ne 100 -c 256 -error 0.05 0.10 0.15 -svptype rapsavgerrorctrl -lm 0.5 -kreg 5 --rand --no-hm |& tee ./logs/exp_f_raps_random_l05_k5_cal256.txt 

#python -u experiments.py -p /home/data/tfmortier/Research/Datasets/CAL256 -k 256 -dim 200 -hi 1000 -b 32 -ne 100 -c 1 256 -error 0.05 0.10 0.15 -svptype avgerrorctrl |& tee ./logs/exp_h_noraps_cal256.txt
#python -u experiments.py -p /home/data/tfmortier/Research/Datasets/CAL256 -k 256 -dim 200 -hi 1000 -b 32 -ne 100 -c 256 -error 0.05 0.10 0.15 -svptype avgerrorctrl --no-hm |& tee ./logs/exp_f_noraps_cal256.txt

# PLANTCLEF
#python -u experiments.py -p /home/data/tfmortier/Research/Datasets/PLANTCLEF -k 1000 -dim 200 -hi 1000 -b 32 -ne 100 -c 1 1000 -error 0.05 0.10 0.15 -svptype rapsavgerrorctrl -lm 0.0 --rand |& tee ./logs/exp_h_raps_random_l0_plantclef.txt
#python -u experiments.py -p /home/data/tfmortier/Research/Datasets/PLANTCLEF -k 1000 -dim 200 -hi 1000 -b 32 -ne 100 -c 1000 -error 0.05 0.10 0.15 -svptype rapsavgerrorctrl -lm 0.0 --rand --no-hm |& tee ./logs/exp_f_raps_random_l0_plantclef.txt
#
#python -u experiments.py -p /home/data/tfmortier/Research/Datasets/PLANTCLEF -k 1000 -dim 200 -hi 1000 -b 32 -ne 100 -c 1 1000 -error 0.05 0.10 0.15 -svptype rapsavgerrorctrl -lm 0.0 --no-rand |& tee ./logs/exp_h_raps_nonrandom_l0_plantclef.txt
#python -u experiments.py -p /home/data/tfmortier/Research/Datasets/PLANTCLEF -k 1000 -dim 200 -hi 1000 -b 32 -ne 100 -c 1000 -error 0.05 0.10 0.15 -svptype rapsavgerrorctrl -lm 0.0 --no-rand --no-hm |& tee ./logs/exp_f_raps_nonrandom_l0_plantclef.txt 
#
#python -u experiments.py -p /home/data/tfmortier/Research/Datasets/PLANTCLEF -k 1000 -dim 200 -hi 1000 -b 32 -ne 100 -c 1 1000 -error 0.05 0.10 0.15 -svptype rapsavgerrorctrl -lm 0.5 -kreg 5 --rand |& tee ./logs/exp_h_raps_random_l05_k5_plantclef.txt
#python -u experiments.py -p /home/data/tfmortier/Research/Datasets/PLANTCLEF -k 1000 -dim 200 -hi 1000 -b 32 -ne 100 -c 1000 -error 0.05 0.10 0.15 -svptype rapsavgerrorctrl -lm 0.5 -kreg 5 --rand --no-hm |& tee ./logs/exp_f_raps_random_l05_k5_plantclef.txt 

#python -u experiments.py -p /home/data/tfmortier/Research/Datasets/PLANTCLEF -k 1000 -dim 200 -hi 1000 -b 32 -ne 100 -c 1 1000 -error 0.05 0.10 0.15 -svptype avgerrorctrl |& tee ./logs/exp_h_noraps_plantclef.txt
#python -u experiments.py -p /home/data/tfmortier/Research/Datasets/PLANTCLEF -k 1000 -dim 200 -hi 1000 -b 32 -ne 100 -c 1000 -error 0.05 0.10 0.15 -svptype avgerrorctrl --no-hm |& tee ./logs/exp_f_noraps_plantclef.txt


#python -u experiments.py -p /home/data/tfmortier/Research/Datasets/PLANTCLEF -k 1000 -dim 200 -hi 1000 -l 0.0001 -b 32 -ne 100 -c 1 1000 -error 0.05 0.10 0.15 -svptype avgerrorctrl |& tee ./logs/exp_h_noaps_plantclef.txt
#python -u experiments.py -p /home/data/tfmortier/Research/Datasets/PLANTCLEF -k 1000 -dim 200 -hi 1000 -l 0.0001 -b 32 -ne 100 -c 1 1000 -error 0.05 0.10 0.15 -svptype rapsavgerrorctrl |& tee ./logs/exp_h_aps_plantclef.txt
#
#python -u experiments.py -p /home/data/tfmortier/Research/Datasets/PLANTCLEF -k 1000 -dim 200 --no-hm -hi 1000 -l 0.0001 -b 32 -ne 100 -c 1000 -error 0.05 0.10 0.15 -svptype avgerrorctrl |& tee ./logs/exp_f_noaps_plantclef.txt
#python -u experiments.py -p /home/data/tfmortier/Research/Datasets/PLANTCLEF -k 1000 -dim 200 --no-hm -hi 1000 -l 0.0001 -b 32 -ne 100 -c 1000 -error 0.05 0.10 0.15 -svptype rapsavgerrorctrl |& tee ./logs/exp_f_aps_plantclef.txt
#
#python -u experiments.py -p /home/data/tfmortier/Research/Datasets/PLANTCLEF --rh -k 1000 -dim 200 -hi 1000 -l 0.0001 -b 32 -ne 100 -c 1 1000 -error 0.05 0.10 0.15 -svptype avgerrorctrl |& tee ./logs/exp_rh_noaps_plantclef.txt
#python -u experiments.py -p /home/data/tfmortier/Research/Datasets/PLANTCLEF --rh -k 1000 -dim 200 -hi 1000 -l 0.0001 -b 32 -ne 100 -c 1 1000 -error 0.05 0.10 0.15 -svptype rapsavgerrorctrl |& tee ./logs/exp_rh_aps_plantclef.txt

# New experiments 17/06/2024

# CAL101 
#python -u experiments.py -p /home/data/tfmortier/Research/Datasets/CAL101 -k 97 -dim 200 -hi 1000 -b 32 -ne 100 -c 1 97 -error 0.05 0.10 0.15 |& tee ./logs/exp_h_cal101.txt
#python -u experiments.py -p /home/data/tfmortier/Research/Datasets/CAL101 -k 97 -dim 200 --no-hm -hi 1000 -b 32 -ne 100 -c 97 -error 0.05 0.10 0.15 |& tee ./logs/exp_f_cal101.txt
#python -u experiments.py -p /home/data/tfmortier/Research/Datasets/CAL101 --rh -k 97 -dim 200 -hi 1000 -b 32 -ne 100 -c 1 97 -error 0.05 0.10 0.15 |& tee ./logs/exp_rh_cal101.txt

# CAL256 (debug)
#python -u experiments.py -p /home/data/tfmortier/Research/Datasets/CAL256 -k 256 -dim 200 -hi 1000 -b 32 -ne 100 -c 1 256 -error 0.05 0.10 0.15 -svptype avgerrorctrl |& tee ./logs/exp_h_noaps_cal256.txt
#python -u experiments.py -p /home/data/tfmortier/Research/Datasets/CAL256 -k 256 -dim 200 -hi 1000 -b 32 -ne 100 -c 1 256 -error 0.05 0.10 0.15 -svptype apsavgerrorctrl |& tee ./logs/exp_h_aps_cal256.txt
#
#python -u experiments.py -p /home/data/tfmortier/Research/Datasets/CAL256 -k 256 -dim 200 --no-hm -hi 1000 -b 32 -ne 100 -c 256 -error 0.05 0.10 0.15 -svptype avgerrorctrl -out avgerrorctrl |& tee ./logs/exp_f_noaps_cal256_debug.txt
#python -u experiments.py -p /home/data/tfmortier/Research/Datasets/CAL256 -k 256 -dim 200 --no-hm -hi 1000 -b 32 -ne 100 -c 256 -error 0.05 0.10 0.15 -svptype apsavgerrorctrl -out apsavgerrorctrl |& tee ./logs/exp_f_aps_cal256_debug.txt

#python -u experiments.py -p /home/data/tfmortier/Research/Datasets/CAL256 -k 256 -dim 200 -hi 1000 -b 32 -ne 100 -c 256 -error 0.05 -svptype apsavgerrorctrl |& tee ./logs/exp_h_aps_cal256_debug.txt

## CAL256
#python -u experiments.py -p /home/data/tfmortier/Research/Datasets/CAL256 -k 256 -dim 200 -hi 1000 -b 32 -ne 100 -c 1 256 -error 0.05 0.10 0.15 -svptype avgerrorctrl |& tee ./logs/exp_h_noaps_cal256.txt
#python -u experiments.py -p /home/data/tfmortier/Research/Datasets/CAL256 -k 256 -dim 200 -hi 1000 -b 32 -ne 100 -c 1 256 -error 0.05 0.10 0.15 -svptype rapsavgerrorctrl |& tee ./logs/exp_h_aps_cal256.txt
#
#python -u experiments.py -p /home/data/tfmortier/Research/Datasets/CAL256 -k 256 -dim 200 --no-hm -hi 1000 -b 32 -ne 100 -c 256 -error 0.05 0.10 0.15 -svptype avgerrorctrl |& tee ./logs/exp_f_noaps_cal256.txt
#python -u experiments.py -p /home/data/tfmortier/Research/Datasets/CAL256 -k 256 -dim 200 --no-hm -hi 1000 -b 32 -ne 100 -c 256 -error 0.05 0.10 0.15 -svptype rapsavgerrorctrl |& tee ./logs/exp_f_aps_cal256.txt
#
#python -u experiments.py -p /home/data/tfmortier/Research/Datasets/CAL256 --rh -k 256 -dim 200 -hi 1000 -b 32 -ne 100 -c 1 256 -error 0.05 0.10 0.15 -svptype avgerrorctrl |& tee ./logs/exp_rh_noaps_cal256.txt
#python -u experiments.py -p /home/data/tfmortier/Research/Datasets/CAL256 --rh -k 256 -dim 200 -hi 1000 -b 32 -ne 100 -c 1 256 -error 0.05 0.10 0.15 -svptype rapsavgerrorctrl |& tee ./logs/exp_rh_aps_cal256.txt


## PlantCLEF2015
#python -u experiments.py -p /home/data/tfmortier/Research/Datasets/PLANTCLEF -k 1000 -dim 200 -hi 1000 -l 0.0001 -b 32 -ne 100 -c 1 1000 -error 0.05 0.10 0.15 -svptype avgerrorctrl |& tee ./logs/exp_h_noaps_plantclef.txt
#python -u experiments.py -p /home/data/tfmortier/Research/Datasets/PLANTCLEF -k 1000 -dim 200 -hi 1000 -l 0.0001 -b 32 -ne 100 -c 1 1000 -error 0.05 0.10 0.15 -svptype rapsavgerrorctrl |& tee ./logs/exp_h_aps_plantclef.txt
#
#python -u experiments.py -p /home/data/tfmortier/Research/Datasets/PLANTCLEF -k 1000 -dim 200 --no-hm -hi 1000 -l 0.0001 -b 32 -ne 100 -c 1000 -error 0.05 0.10 0.15 -svptype avgerrorctrl |& tee ./logs/exp_f_noaps_plantclef.txt
#python -u experiments.py -p /home/data/tfmortier/Research/Datasets/PLANTCLEF -k 1000 -dim 200 --no-hm -hi 1000 -l 0.0001 -b 32 -ne 100 -c 1000 -error 0.05 0.10 0.15 -svptype rapsavgerrorctrl |& tee ./logs/exp_f_aps_plantclef.txt
#
#python -u experiments.py -p /home/data/tfmortier/Research/Datasets/PLANTCLEF --rh -k 1000 -dim 200 -hi 1000 -l 0.0001 -b 32 -ne 100 -c 1 1000 -error 0.05 0.10 0.15 -svptype avgerrorctrl |& tee ./logs/exp_rh_noaps_plantclef.txt
#python -u experiments.py -p /home/data/tfmortier/Research/Datasets/PLANTCLEF --rh -k 1000 -dim 200 -hi 1000 -l 0.0001 -b 32 -ne 100 -c 1 1000 -error 0.05 0.10 0.15 -svptype rapsavgerrorctrl |& tee ./logs/exp_rh_aps_plantclef.txt

## BACT
#python -u experiments.py -p /home/data/tfmortier/Research/Datasets/BACT -k 2659 -dim 2457 -hi 1000 -l 0.01 -b 32 -ne 100 -c 1 2659 -error 0.05 0.10 0.15 |& tee ./logs/exp_h_bact.txt
#python -u experiments.py -p /home/data/tfmortier/Research/Datasets/BACT -k 2659 -dim 2457 --no-hm -hi 1000 -l 0.01 -b 32 -ne 100 -c 2659 -error 0.05 0.10 0.15 |& tee ./logs/exp_f_bact.txt
#python -u experiments.py -p /home/data/tfmortier/Research/Datasets/BACT --rh -k 2659 -dim 2457 -hi 1000 -l 0.01 -b 32 -ne 100 -c 1 2659 -error 0.05 0.10 0.15 |& tee ./logs/exp_rh_bact.txt

## PROT
#python -u experiments.py -p /home/data/tfmortier/Research/Datasets/PROT -k 3485 -dim 26276 -hi 1000 -l 0.01 -b 32 -ne 100 -c 1 3485 -error 0.05 0.10 0.15 -svptype avgerrorctrl |& tee ./logs/exp_h_noaps_prot.txt
#python -u experiments.py -p /home/data/tfmortier/Research/Datasets/PROT -k 3485 -dim 26276 -hi 1000 -l 0.01 -b 32 -ne 100 -c 1 3485 -error 0.05 0.10 0.15 -svptype rapsavgerrorctrl |& tee ./logs/exp_h_aps_prot.txt
#
#python -u experiments.py -p /home/data/tfmortier/Research/Datasets/PROT -k 3485 -dim 26276 --no-hm -hi 1000 -l 0.01 -b 32 -ne 100 -c 3485 -error 0.05 0.10 0.15 -svptype avgerrorctrl |& tee ./logs/exp_f_noaps_prot.txt
#python -u experiments.py -p /home/data/tfmortier/Research/Datasets/PROT -k 3485 -dim 26276 --no-hm -hi 1000 -l 0.01 -b 32 -ne 100 -c 3485 -error 0.05 0.10 0.15 -svptype rapsavgerrorctrl |& tee ./logs/exp_f_aps_prot.txt
#
#python -u experiments.py -p /home/data/tfmortier/Research/Datasets/PROT --rh -k 3485 -dim 26276 -hi 1000 -l 0.01 -b 32 -ne 100 -c 1 3485 -error 0.05 0.10 0.15 -svptype avgerrorctrl |& tee ./logs/exp_rh_noaps_prot.txt
#python -u experiments.py -p /home/data/tfmortier/Research/Datasets/PROT --rh -k 3485 -dim 26276 -hi 1000 -l 0.01 -b 32 -ne 100 -c 1 3485 -error 0.05 0.10 0.15 -svptype rapsavgerrorctrl |& tee ./logs/exp_rh_aps_prot.txt

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
