# run as for i in `seq 0 62`; do; nohup python haririhammer_calculate_erf.py $i > log/erf_$i.log; done
# Calculate ERFs
import numpy as np
import mne
import sys
import os
from os.path import join as pathjoin
from erf_acw.src import get_commonsubj, pklsave, get_subjdatapath

taskname = "haririhammer"

subjs_common = get_commonsubj()

i = int(sys.argv[1])
i_subj = subjs_common[i]

##################################################

nsubj = len(subjs_common)

print(f"Starting subj: {i_subj}")
preprocpath = "/BICNAS2/group-northoff/NIMH_healthy_volunteer/preprocessing"
subj_preprocpath = pathjoin(preprocpath, i_subj)
outputpath = pathjoin(subj_preprocpath, "haririhammer", "rejection_ica")

filename = pathjoin(
    outputpath, i_subj + "_haririhammer_preprocessed_erf_emo-epo.fif"
)

tasknames = ["encode_face_happy", "probe_face_happy",
             "encode_face_sad", "probe_face_sad",
             "encode_shape", "probe_shape"]
event_id = {i_task: i+1 for i, i_task in enumerate(tasknames)}

def nanmean(data):
    return np.nanmean(data, axis=0)

epochs = mne.read_epochs(filename)
probe_face_ave_happy = epochs["probe_face_happy"].average(method=nanmean)
encode_face_ave_happy = epochs["encode_face_happy"].average(method=nanmean)
probe_face_ave_sad = epochs["probe_face_sad"].average(method=nanmean)
encode_face_ave_sad = epochs["encode_face_sad"].average(method=nanmean)
probe_shape_ave = epochs["probe_shape"].average(method=nanmean)
encode_shape_ave = epochs["encode_shape"].average(method=nanmean)

filename = pathjoin(
    "/BICNAS2/ycatal/erf_acw2/results/erf/haririhammer",
    f"{i_subj}_{taskname}_erf_emo.pkl"
)

pklsave(filename, {"encode_face_happy": encode_face_ave_happy, 
                   "probe_face_happy": probe_face_ave_happy, 
                   "encode_face_sad": encode_face_ave_sad, 
                   "probe_face_sad": probe_face_ave_sad, 
                   "probe_shape": probe_shape_ave,
                   "encode_shape": encode_shape_ave, 
                   })

print("done")


