#! /bin/bash

cd /BICNAS2/ycatal/erf_acw2/scripts/preprocessing/source_reconstruction

for i in {51..60}
do
    nohup python SX_5a3_inverse_solution_haririhammer_prestim_lcmv.py $i > log/SX_5a3_inverse_solution_haririhammer_prestim_lcmv_$i.log &
    echo $! >> log/SX_5a3_inverse_solution_haririhammer_prestim_lcmv.pid
done

# In case something goes wrong, kill the processes with:
# kill -9 $(cat log/SX_1_process_empty_room.pid)
# rm log/SX_1_process_empty_room.pid
# rm log/*