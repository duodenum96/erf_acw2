#! /bin/bash

cd /BICNAS2/ycatal/erf_acw2/scripts/preprocessing/source_reconstruction

for i in {31..60}
do
    nohup python SX_3b_alignment_rest.py $i > log/SX_3b_alignment_rest_$i.log &
    echo $! >> log/SX_3b_alignment_rest.pid
done

# In case something goes wrong, kill the processes with:
# kill -9 $(cat log/SX_1_process_empty_room.pid)
# rm log/SX_1_process_empty_room.pid
# rm log/*