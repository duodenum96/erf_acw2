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
    "/BICNAS2/ycatal/erf_acw2/results/erf/haririhammer",
    f"{i_subj}_{taskname}_erf_emo_induced.pkl"
)


tasknames = ["encode_face_happy", "probe_face_happy",
             "encode_face_sad", "probe_face_sad",
             "encode_shape", "probe_shape"]
event_id = {i_task: i+1 for i, i_task in enumerate(tasknames)}

induced = pklload(filename)
induced_keys = list(induced.keys())

freqs = np.logspace(*np.log10([6, 50]), num=20)
n_cycles = freqs / 2.0  # different number of cycle per frequency
induced_itpc = {}
induced_power = {}

for key in induced_keys:
    induced_power[key], induced_itpc[key] = induced[key].compute_tfr(
        method="morlet",
        freqs=freqs,
        n_cycles=n_cycles,
        average=True,
        return_itc=True,
        decim=8,
    )
    # induced_power[key].apply_baseline(mode="ratio", baseline=(None, 0))
    

filename = pathjoin(
    "/BICNAS2/ycatal/erf_acw2/results/erf/haririhammer",
    f"{i_subj}_{taskname}_erf_emo_induced_power_no_baseline.pkl"
)

pklsave(filename, {"induced_power": induced_power, "induced_itpc": induced_itpc})

print(f"Finished subj: {i_subj}")

