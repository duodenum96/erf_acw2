#!/bin/bash
cd /BICNAS2/ycatal/erf_acw2/scripts/acw

export OMP_NUM_THREADS=2

for i in {41..60}; do
    nohup python S13_X_calculate_acw_oscillatory_fit.py $i > log/S13_X_calculate_acw_oscillatory_fit_$i.log &
    echo $! >> log/oscillatory_fit_pids.txt
done

# Cancel the jobs if things go wrong
while read line; do
    kill $line
done < log/oscillatory_fit_pids.txt