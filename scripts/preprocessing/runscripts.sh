#!/bin/bash
# The scripts in the folder can be run with the following example code for parallel processing
cd /BICNAS2/ycatal/erf_acw2/scripts/preprocessing

for subj in sub-ON72082 \
sub-ON82386 \
sub-ON42107 \
sub-ON61373 \
sub-ON95422 \
sub-ON22671 \
sub-ON85305
do
nohup python haririhammer_preproc_1_continuous.py $subj > log/${subj}_haririhammer_1_continuous.log &
echo $! >> log/pids.txt
done

