#! /bin/bash

cd /BICNAS2/ycatal/erf_acw2/scripts/erf_calculation
for i_subj in {31..60}
do
    nohup python SZ1_calculate_induced.py $i_subj > log/SZ1_calculate_induced_$i_subj.log &
    echo $! >> log/SZ1_calculate_induced.pid
done

