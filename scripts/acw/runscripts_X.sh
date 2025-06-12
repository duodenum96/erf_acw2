#!/bin/bash
cd /BICNAS2/ycatal/erf_acw2/scripts/acw

export OMP_NUM_THREADS=2

for i in {0..20}; do
    nohup python S13_X_calculate_acw_FOOOF.py $i > log/S13_X_calculate_acw_FOOOF_$i.log &
    echo $! >> log/fooof_pids.txt
done

# Cancel the jobs if things go wrong
while read line; do
    kill $line
done < log/fooof_pids.txt