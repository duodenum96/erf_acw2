#! /bin/bash

cd /BICNAS2/ycatal/erf_acw2/scripts/preprocessing/source_reconstruction

for i in {31..60}
do
    nohup python SX_2_calculate_noise_covariance.py $i > log/SX_2_calculate_noise_covariance_$i.log &
    echo $! >> log/SX_2_calculate_noise_covariance.pid
done

# In case something goes wrong, kill the processes with:
# kill -9 $(cat log/SX_1_process_empty_room.pid)
# rm log/SX_1_process_empty_room.pid
# rm log/*