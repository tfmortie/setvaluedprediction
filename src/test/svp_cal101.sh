#!/bin/bash

# Fbeta
python -u svp_cal101.py -ne 3 -np 200 --no-hm --gpu -constraint none -beta 1 -c 1 |& tee nohm_b1_1_cal101.txt
python -u svp_cal101.py -ne 3 -np 200 --no-hm --gpu -constraint none -beta 1 -c 2 |& tee nohm_b1_2_cal101.txt
python -u svp_cal101.py -ne 3 -np 200 --no-hm --gpu -constraint none -beta 1 -c 3 |& tee nohm_b1_3_cal101.txt
python -u svp_cal101.py -ne 3 -np 200 --no-hm --gpu -constraint none -beta 1 -c 101 |& tee nohm_b1_101_cal101.txt

python -u svp_cal101.py -ne 3 -np 200 --hm --gpu -constraint none -beta 1 -c 1 |& tee hm_b1_1_cal101.txt
python -u svp_cal101.py -ne 3 -np 200 --hm --gpu -constraint none -beta 1 -c 2 |& tee hm_b1_2_cal101.txt
python -u svp_cal101.py -ne 3 -np 200 --hm --gpu -constraint none -beta 1 -c 101 |& tee hm_b1_101_cal101.txt

# Size restriction
python -u svp_cal101.py -ne 3 -np 200 --no-hm --gpu -constraint size -size 5 -c 1 |& tee nohm_k5_1_cal101.txt
python -u svp_cal101.py -ne 3 -np 200 --no-hm --gpu -constraint size -size 5 -c 2 |& tee nohm_k5_2_cal101.txt
python -u svp_cal101.py -ne 3 -np 200 --no-hm --gpu -constraint size -size 5 -c 3 |& tee nohm_k5_3_cal101.txt
python -u svp_cal101.py -ne 3 -np 200 --no-hm --gpu -constraint size -size 5 -c 101 |& tee nohm_k5_101_cal101.txt

python -u svp_cal101.py -ne 3 -np 200 --hm --gpu -constraint size -size 5 -c 1 |& tee hm_k5_1_cal101.txt
python -u svp_cal101.py -ne 3 -np 200 --hm --gpu -constraint size -size 5 -c 2 |& tee hm_k5_2_cal101.txt
python -u svp_cal101.py -ne 3 -np 200 --hm --gpu -constraint size -size 5 -c 3 |& tee hm_k5_3_cal101.txt
python -u svp_cal101.py -ne 3 -np 200 --hm --gpu -constraint size -size 5 -c 101 |& tee hm_k5_101_cal101.txt

# Error restriction
python -u svp_cal101.py -ne 3 -np 200 --no-hm --gpu -constraint error -error 0.10 -c 1 |& tee nohm_e10_1_cal101.txt
python -u svp_cal101.py -ne 3 -np 200 --no-hm --gpu -constraint error -error 0.10 -c 2 |& tee nohm_e10_2_cal101.txt
python -u svp_cal101.py -ne 3 -np 200 --no-hm --gpu -constraint error -error 0.10 -c 3 |& tee nohm_e10_3_cal101.txt
python -u svp_cal101.py -ne 3 -np 200 --no-hm --gpu -constraint error -error 0.10 -c 101 |& tee nohm_e10_101_cal101.txt

python -u svp_cal101.py -ne 3 -np 200 --hm --gpu -constraint error -error 0.10 -c 1 |& tee hm_e10_1_cal101.txt
python -u svp_cal101.py -ne 3 -np 200 --hm --gpu -constraint error -error 0.10 -c 2 |& tee hm_e10_2_cal101.txt
python -u svp_cal101.py -ne 3 -np 200 --hm --gpu -constraint error -error 0.10 -c 3 |& tee hm_e10_3_cal101.txt
python -u svp_cal101.py -ne 3 -np 200 --hm --gpu -constraint error -error 0.10 -c 101 |& tee hm_e10_101_cal101.txt
