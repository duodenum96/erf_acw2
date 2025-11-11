# run as for i in `seq 0 62`; do; nohup python haririhammer_calculate_erf.py $i > log/erf_$i.log; done
# Calculate ERFs
import numpy as np
import mne
import sys
import os
from os.path import join as pathjoin
from erf_acw2.src import get_commonsubj, pklsave, pklload

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

evoked_filename = pathjoin(
    "/BICNAS2/ycatal/erf_acw2/results/erf/haririhammer",
    f"{i_subj}_{taskname}_erf_emo.pkl"
)

evokeds = pklload(evoked_filename)

encode_face_happy = evokeds["encode_face_happy"]
probe_face_happy = evokeds["probe_face_happy"]
encode_face_sad = evokeds["encode_face_sad"]
probe_face_sad = evokeds["probe_face_sad"]
probe_shape = evokeds["probe_shape"]
encode_shape = evokeds["encode_shape"]

induced_encode_face_happy = epochs["encode_face_happy"].subtract_evoked(encode_face_happy)
induced_probe_face_happy = epochs["probe_face_happy"].subtract_evoked(probe_face_happy)
induced_encode_face_sad = epochs["encode_face_sad"].subtract_evoked(encode_face_sad)
induced_probe_face_sad = epochs["probe_face_sad"].subtract_evoked(probe_face_sad)
induced_probe_shape = epochs["probe_shape"].subtract_evoked(probe_shape)
induced_encode_shape = epochs["encode_shape"].subtract_evoked(encode_shape)


filename = pathjoin(
    "/BICNAS2/ycatal/erf_acw2/results/erf/haririhammer",
    f"{i_subj}_{taskname}_erf_emo_induced.pkl"
)

pklsave(filename, {"encode_face_happy": induced_encode_face_happy, 
                   "probe_face_happy": induced_probe_face_happy, 
                   "encode_face_sad": induced_encode_face_sad, 
                   "probe_face_sad": induced_probe_face_sad, 
                   "probe_shape": induced_probe_shape,
                   "encode_shape": induced_encode_shape, 
                   })

print("done")




