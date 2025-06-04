#!/bin/bash

cd /BICNAS2/ycatal/erf_acw2/scripts/preprocessing/source_reconstruction

for i in {0..20}; do
    nohup bash SX_0_run_freesurfer.sh $i > log/freesurfer_${i}.log 2>&1 &
    echo $! >> log/freesurfer_pids.txt
done

