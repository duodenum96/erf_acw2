#! /bin/bash

cd /BICNAS2/ycatal/erf_acw2/scripts/preprocessing/source_reconstruction

for i in {51..60}
do
    nohup python SX_5b_inverse_solution_rest_acf.py $i > log/SX_5b_inverse_solution_rest_acf_$i.log &
    echo $! >> log/SX_5b_inverse_solution_rest_acf.pid
done

# In case something goes wrong, kill the processes with:
# kill -9 $(cat log/SX_5b_inverse_solution_rest.pid)
# rm log/SX_5b_inverse_solution_rest.pid
# rm log/*

# 7, 9, 11, 12, 15, 19, 20 