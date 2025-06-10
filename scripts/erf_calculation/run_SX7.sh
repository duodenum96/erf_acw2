#! /bin/bash

cd /BICNAS2/ycatal/erf_acw2/scripts/erf_calculation
nohup python SX7_haririhammer_st_permutation_source.py > log/SX7_haririhammer_st_permutation_source.log &
echo $! > log/SX7_haririhammer_st_permutation_source.pid

