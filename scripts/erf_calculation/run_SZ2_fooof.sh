#! /bin/bash

cd /BICNAS2/ycatal/erf_acw2/scripts/erf_calculation
for i_subj in {41..60}
do
    nohup python SZ2_calculate_induced_power_fooof.py $i_subj > log/SZ2_calculate_induced_power_fooof_$i_subj.log &
    echo $! >> log/SZ2_calculate_induced_power_fooof.pid
done

