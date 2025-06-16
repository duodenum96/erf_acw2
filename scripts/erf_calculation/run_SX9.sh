#! /bin/bash

cd /BICNAS2/ycatal/erf_acw2/scripts/erf_calculation
nohup python SX9_compile_grand_average_for_clusters.py > log/SX9_compile_grand_average_for_clusters.log &
echo $! > log/SX9_compile_grand_average_for_clusters.pid

