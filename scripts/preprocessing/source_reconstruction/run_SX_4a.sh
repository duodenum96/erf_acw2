#! /bin/bash

cd /BICNAS2/ycatal/erf_acw2/scripts/preprocessing/source_reconstruction

for i in {41..60}
do
    nohup python SX_4a_forward_computation_haririhammer_volume.py $i > log/SX_4a_forward_computation_haririhammer_volume_$i.log &
    echo $! >> log/SX_4a_forward_computation_haririhammer_volume.pid
done

# In case something goes wrong, kill the processes with:
# kill -9 $(cat log/SX_1_process_empty_room.pid)
# rm log/SX_1_process_empty_room.pid
# rm log/*