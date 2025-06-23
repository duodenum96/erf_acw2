#! /bin/bash

cd /BICNAS2/ycatal/erf_acw2/scripts/acw

for i in {41..60}
do
    nohup python SX11_calculate_acw_FOOOF.py $i > log/SX11_calculate_acw_FOOOF_$i.log &
    echo $! >> log/SX11_calculate_acw_FOOOF.pid
done

# In case something goes wrong, kill the processes with:
# kill -9 $(cat log/SX11_calculate_acw_FOOOF.pid)
# rm log/SX11_calculate_acw_FOOOF.pid
# rm log/*

# 