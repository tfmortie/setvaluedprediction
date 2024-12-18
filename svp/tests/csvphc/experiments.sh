#!/bin/bash
#
# Experiments for "Distribution-free set-valued prediction in hierarchical classification with finite sample guarantees"
#
# Author: Thomas Mortier
# Date: September 2024
#

# Experiments 26/11/2024

#python -u experiments.py -p /home/data/tfmortier/Research/Datasets/PLANTCLEF -out DEBUGPLANTCLEF2712 --config ./experiments/ampsr.yaml ./experiments/apsr.yaml -nexp 20 -k 1000 -dim 200 -hi 1000 -b 32 -ne 100 |& tee ./logs/plantclef_171224.txt
#python -u experiments.py -p /home/data/tfmortier/Research/Datasets/PLANTCLEF -out DEBUGPLANTCLEF2712 --config ./experiments/apsr.yaml -nexp 20 -k 1000 -dim 200 -hi 1000 -b 32 -ne 2 |& tee ./logs/plantclef_171224.txt
#python -u experiments.py -p /home/data/tfmortier/Research/Datasets/CAL101 -out DEBUGCAL101 --config ./experiments/apsr.yaml -np 50 -nexp 20 -k 97 -dim 200 -hi 1000 -b 32 -ne 5 |& tee ./logs/cal101_181224.txt

# Experiments 18/12/2024

# DEBUG
#python -u experiments.py -p /home/data/tfmortier/Research/Datasets/CAL256 --config ./experiments/apsrdebug.yaml -nexp 1 -k 256 -dim 200 -hi 1000 -b 32 -ne 10 |& tee ./logs/cal256_debug_issue.txt

# CIFAR10
#python -u experiments.py -p /home/data/tfmortier/Research/Datasets/CIFAR10 --config ./experiments/apsrdebug.yaml -nexp 1 -k 10 -dim 200 -hi 1000 -b 32 -ne 100 |& tee ./logs/debug.txt
python -u experiments.py -p /home/data/tfmortier/Research/Datasets/CIFAR10 --config ./experiments/ -nexp 20 -k 10 -dim 200 -hi 1000 -b 32 -ne 100 |& tee ./logs/cifar10.txt

# CAL101
python -u experiments.py -p /home/data/tfmortier/Research/Datasets/CAL101 --config ./experiments/ -np 50 -nexp 20 -k 97 -dim 200 -hi 1000 -b 32 -ne 100 |& tee ./logs/cal101.txt

# CAL256
python -u experiments.py -p /home/data/tfmortier/Research/Datasets/CAL256 --config ./experiments/ -nexp 20 -k 256 -dim 200 -hi 1000 -b 32 -ne 100 |& tee ./logs/cal256.txt

# PLANTCLEF
python -u experiments.py -p /home/data/tfmortier/Research/Datasets/PLANTCLEF --config ./experiments/ -nexp 20 -k 1000 -dim 200 -hi 1000 -b 32 -ne 100 |& tee ./logs/plantclef.txt

# BACTERIA
python -u experiments.py -p /home/data/tfmortier/Research/Datasets/BACT --config ./experiments -nexp 20 -k 2659 -dim 2457 -hi 1000 -b 32 -l 0.01 -ne 100 |& tee ./logs/bact.txt

# PROT
python -u experiments.py -p /home/data/tfmortier/Research/Datasets/PROT --config ./experiments/ -nexp 20 -k 3485 -dim 26276 -hi 1000 -b 32 -l 0.01 -ne 100 |& tee ./logs/prot.txt


# Additional experiments
# CAL101
#python -u experiments.py -p /home/data/tfmortier/Research/Datasets/CAL101 --config ./experiments/ -nexp 50 -k 97 -dim 200 -hi 1000 -b 32 -ne 100 |& tee ./logs/cal101.txt

# PROT
#python -u experiments.py -p /home/data/tfmortier/Research/Datasets/PROT --config ./experiments/ -nexp 50 -k 3485 -dim 26276 -hi 1000 -b 32 -l 0.01 -ne 100 |& tee ./logs/prot.txt

# OLD

# CIFAR10
# Debug
#python -u experiments.py -p /home/data/tfmortier/Research/Datasets/CIFAR10 --config ./experiments/rapsusvphfrandom.yaml ./experiments/rapsusvphfrandomreg.yaml -nexp 1 -k 10 -dim 200 -hi 1000 -b 32 -ne 2 |& tee ./logs/testje.txt
#python -u experiments.py -p /home/data/tfmortier/Research/Datasets/CIFAR10 --config ./experiments/crsvphfrsvphf.yaml ./experiments/crsvphfrsvphfrandom.yaml ./experiments/csvphfrsvphf.yaml ./experiments/rapsusvphf.yaml ./experiments/rapsusvphfrandom.yaml -nexp 1 -k 10 -dim 200 -hi 1000 -b 32 -ne 2 |& tee ./logs/testje.txt
#python -u experiments.py -p /home/data/tfmortier/Research/Datasets/CIFAR10 --config ./experiments/rapsusvphfrandom.yaml -nexp 1 -k 10 -dim 200 -hi 1000 -b 32 -ne 2 |& tee ./logs/testje.txt
#python -u experiments.py -p /home/data/tfmortier/Research/Datasets/CIFAR10 --config ./experiments/rapsusvphf.yaml ./experiments/rapsusvphf.yaml -nexp 1 -k 10 -dim 200 -hi 1000 -b 32 -ne 2 |& tee ./logs/testje.txt
#python -u experiments.py -p /home/data/tfmortier/Research/Datasets/CIFAR10 --config ./experiments/csvphfrsvphfrandom.yaml ./experiments/csvphfrsvphf.yaml  -nexp 1 -k 10 -dim 200 -hi 1000 -b 32 -ne 2 |& tee ./logs/testje.txt

# New experiments 19/09/2024

# CIFAR10
#python -u experiments.py -p /home/data/tfmortier/Research/Datasets/CIFAR10 -out CIFAR10LAC1 -nexp 10 -k 10 -dim 200 -hi 1000 -b 32 -ne 100 -c 1 -error 0.10 -svptype lac |& tee ./logs/exp_h_lac_1_cifar10.txt
#python -u experiments.py -p /home/data/tfmortier/Research/Datasets/CIFAR10 -out CIFAR10RAPS1 -nexp 10 -k 10 -dim 200 -hi 1000 -b 32 -ne 100 -c 1 -error 0.10 -svptype raps --rand |& tee ./logs/exp_h_raps_1_cifar10.txt
#python -u experiments.py -p /home/data/tfmortier/Research/Datasets/CIFAR10 -out CIFAR10CR1 -nexp 10 -k 10 -dim 200 -hi 1000 -b 32 -ne 100 -c 1 -error 0.10 -svptype crsvphf |& tee ./logs/exp_h_crsvphf_1_cifar10.txt
#python -u experiments.py -p /home/data/tfmortier/Research/Datasets/CIFAR10 -out CIFAR10C1 -nexp 10 -k 10 -dim 200 -hi 1000 -b 32 -ne 2 -c 1 -error 0.10 -svptype csvphf |& tee ./logs/exp_h_csvphf_1_cifar10.txt
#python -u experiments.py -p /home/data/tfmortier/Research/Datasets/CIFAR10 -out CIFAR10C2 -nexp 10 -k 10 -dim 200 -hi 1000 -b 32 -ne 100 -c 2 -error 0.10 -svptype csvphf |& tee ./logs/exp_h_csvphf_2_cifar10.txt
#python -u experiments.py -p /home/data/tfmortier/Research/Datasets/CIFAR10 -out CIFAR10C3 -nexp 10 -k 10 -dim 200 -hi 1000 -b 32 -ne 100 -c 3 -error 0.10 -svptype csvphf |& tee ./logs/exp_h_csvphf_3_cifar10.txt
#
## CALTECH256
#python -u experiments.py -p /home/data/tfmortier/Research/Datasets/CAL256 -out CAL256LAC1 -nexp 10 -k 256 -dim 200 -hi 1000 -b 32 -ne 100 -c 1 -error 0.10 -svptype lac |& tee ./logs/exp_h_lac_1_cal256.txt
#python -u experiments.py -p /home/data/tfmortier/Research/Datasets/CAL256 -out CAL256RAPS1 -nexp 10 -k 256 -dim 200 -hi 1000 -b 32 -ne 100 -c 1 -error 0.10 -svptype raps --rand |& tee ./logs/exp_h_raps_1_cal256.txt
#python -u experiments.py -p /home/data/tfmortier/Research/Datasets/CAL256 -out CAL256CR1 -nexp 10 -k 256 -dim 200 -hi 1000 -b 32 -ne 100 -c 1 -error 0.10 -svptype crsvphf |& tee ./logs/exp_h_crsvphf_1_cal256.txt
#python -u experiments.py -p /home/data/tfmortier/Research/Datasets/CAL256 -out CAL256C1 -nexp 10 -k 256 -dim 200 -hi 1000 -b 32 -ne 100 -c 1 -error 0.10 -svptype csvphf |& tee ./logs/exp_h_csvphf_1_cal256.txt
#python -u experiments.py -p /home/data/tfmortier/Research/Datasets/CAL256 -out CAL256C2 -nexp 10 -k 256 -dim 200 -hi 1000 -b 32 -ne 100 -c 2 -error 0.10 -svptype csvphf |& tee ./logs/exp_h_csvphf_2_cal256.txt
#python -u experiments.py -p /home/data/tfmortier/Research/Datasets/CAL256 -out CAL256C3 -nexp 10 -k 256 -dim 200 -hi 1000 -b 32 -ne 100 -c 3 -error 0.10 -svptype csvphf |& tee ./logs/exp_h_csvphf_3_cal256.txt
#
## PLANTCLEF
#python -u experiments.py -p /home/data/tfmortier/Research/Datasets/PLANTCLEF -out PLANTCLEFLAC1 -nexp 10 -k 1000 -dim 200 -hi 1000 -b 32 -ne 100 -c 1 -error 0.10 -svptype lac |& tee ./logs/exp_h_lac_1_plantclef.txt
#python -u experiments.py -p /home/data/tfmortier/Research/Datasets/PLANTCLEF -out PLANTCLEFRAPS1 -nexp 10 -k 1000 -dim 200 -hi 1000 -b 32 -ne 100 -c 1 -error 0.10 -svptype raps --rand |& tee ./logs/exp_h_raps_1_plantclef.txt
#python -u experiments.py -p /home/data/tfmortier/Research/Datasets/PLANTCLEF -out PLANTCLEFCR1 -nexp 10 -k 1000 -dim 200 -hi 1000 -b 32 -ne 100 -c 1 -error 0.10 -svptype crsvphf |& tee ./logs/exp_h_crsvphf_1_plantclef.txt
#python -u experiments.py -p /home/data/tfmortier/Research/Datasets/PLANTCLEF -out PLANTCLEFC1 -nexp 10 -k 1000 -dim 200 -hi 1000 -b 32 -ne 100 -c 1 -error 0.10 -svptype csvphf |& tee ./logs/exp_h_csvphf_1_plantclef.txt
#python -u experiments.py -p /home/data/tfmortier/Research/Datasets/PLANTCLEF -out PLANTCLEFC2 -nexp 10 -k 1000 -dim 200 -hi 1000 -b 32 -ne 100 -c 2 -error 0.10 -svptype csvphf |& tee ./logs/exp_h_csvphf_2_plantclef.txt
#python -u experiments.py -p /home/data/tfmortier/Research/Datasets/PLANTCLEF -out PLANTCLEFC3 -nexp 10 -k 1000 -dim 200 -hi 1000 -b 32 -ne 100 -c 3 -error 0.10 -svptype csvphf |& tee ./logs/exp_h_csvphf_3_plantclef.txt
#
## BACTERIA
#python -u experiments.py -p /home/data/tfmortier/Research/Datasets/BACT -out BACTLAC1 -l 0.01 -nexp 10 -k 2659 -dim 2457 -hi 1000 -b 32 -ne 100 -c 1 -error 0.10 -svptype lac |& tee ./logs/exp_h_lac_1_bact.txt
#python -u experiments.py -p /home/data/tfmortier/Research/Datasets/BACT -out BACTRAPS1 -l 0.01 -nexp 10 -k 2659 -dim 2457 -hi 1000 -b 32 -ne 100 -c 1 -error 0.10 -svptype raps --rand |& tee ./logs/exp_h_raps_1_bact.txt
#python -u experiments.py -p /home/data/tfmortier/Research/Datasets/BACT -out BACTCR1 -l 0.01 -nexp 10 -k 2659 -dim 2457 -hi 1000 -b 32 -ne 100 -c 1 -error 0.10 -svptype crsvphf |& tee ./logs/exp_h_crsvphf_1_bact.txt
#python -u experiments.py -p /home/data/tfmortier/Research/Datasets/BACT -out BACTC1 -l 0.01 -nexp 10 -k 2659 -dim 2457 -hi 1000 -b 32 -ne 100 -c 1 -error 0.10 -svptype csvphf |& tee ./logs/exp_h_csvphf_1_bact.txt
#python -u experiments.py -p /home/data/tfmortier/Research/Datasets/BACT -out BACTC2 -l 0.01 -nexp 10 -k 2659 -dim 2457 -hi 1000 -b 32 -ne 100 -c 2 -error 0.10 -svptype csvphf |& tee ./logs/exp_h_csvphf_2_bact.txt
#python -u experiments.py -p /home/data/tfmortier/Research/Datasets/BACT -out BACTC3 -l 0.01 -nexp 10 -k 2659 -dim 2457 -hi 1000 -b 32 -ne 100 -c 3 -error 0.10 -svptype csvphf |& tee ./logs/exp_h_csvphf_3_bact.txt
#

#python -u experiments.py -p /home/data/tfmortier/Research/Datasets/BACT --rh -k 2659 -dim 2457 -hi 1000 -l 0.01 -b 32 -ne 100 -c 1 2659 -error 0.05 0.10 0.15 |& tee ./logs/exp_rh_bact.txt

#########################################

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
