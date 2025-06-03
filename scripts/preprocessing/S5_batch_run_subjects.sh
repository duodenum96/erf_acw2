#!/bin/bash

cd /BICNAS2/ycatal/erf_acw2/scripts/preprocessing

for i in {0..4}; do
    nohup bash S5_2_run_freesurfer.sh $i > log/freesurfer_${i}.log 2>&1 &
    echo $! >> log/freesurfer_pids.txt
done

