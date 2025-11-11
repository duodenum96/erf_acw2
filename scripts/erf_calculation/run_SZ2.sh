#! /bin/bash

cd /BICNAS2/ycatal/erf_acw2/scripts/erf_calculation
for i_subj in {31..60}
do
    nohup python SZ2_calculate_induced_power.py $i_subj > log/SZ2_calculate_induced_power_$i_subj.log &
    echo $! >> log/SZ2_calculate_induced_power.pid
done

