#!/bin/bash
cd /BICNAS2/ycatal/erf_acw2/scripts/acw

for i in {31..60}; do
    nohup python S16_calculate_acw.py $i > log/S16_calculate_acw_$i.log &
done
