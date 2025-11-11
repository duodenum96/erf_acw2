import numpy as np
import mne
import sys
import os
from os.path import join as pathjoin
from erf_acw2.src import get_commonsubj, pklsave, pklload
import fooof

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
induced_power_pre = {}
induced_power_post = {}
low_fgs_pre = {}
high_fgs_pre = {}
low_fgs_post = {}
high_fgs_post = {}

low_freqlim_pre = (1, 12)
high_freqlim_pre = (12, 50)
low_freqlim_post = (1, 12)
high_freqlim_post = (12, 50)


for key in induced_keys:
    induced_power_pre[key] = induced[key].average().compute_psd(method="multitaper", tmin=0, fmin=1, fmax=100)
    induced_power_post[key] = induced[key].average().compute_psd(method="multitaper", tmax=0, fmin=1, fmax=100)
    spectra_pre, freqs_pre = induced_power_pre[key].get_data(return_freqs=True)
    low_fg = fooof.FOOOFGroup(max_n_peaks=6)   
    high_fg = fooof.FOOOFGroup(max_n_peaks=6)  
    low_fg.fit(freqs_pre, spectra_pre, low_freqlim_pre) 
    high_fg.fit(freqs_pre, spectra_pre, high_freqlim_pre) 
    low_fgs_pre[key] = low_fg
    high_fgs_pre[key] = high_fg
    

# # check
import matplotlib.pyplot as plt
f, ax = plt.subplots()
induced_power_post[key].plot(xscale="log", axes=ax)
ax.axvline([12])

f.savefig("anan_post.jpg")

filename = pathjoin(
    "/BICNAS2/ycatal/erf_acw2/results/erf/haririhammer",
    f"{i_subj}_{taskname}_erf_emo_induced_fooof.pkl"
)

pklsave(filename, {"induced_power": induced_power, "low_fgs": low_fgs, "high_fgs": high_fgs})

print(f"Finished subj: {i_subj}")

