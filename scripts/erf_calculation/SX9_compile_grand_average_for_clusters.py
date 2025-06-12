import mne
import numpy as np
import os
from os.path import join as pathjoin
import sys
from erf_acw2.src import get_commonsubj

os.chdir("/BICNAS2/ycatal/erf_acw2/scripts/preprocessing")
from scripts.preprocessing.rest_badICs import exclude, badics
from scipy import signal
import matplotlib.pyplot as plt
from erf_acw2.src import pklload, pklsave, badchan_padnan_2d

taskname = "haririhammer"
subjects_dir = "/BICNAS2/group-northoff/NIMH_source_reconstruction"

fsaverage_bem_path = "/BICNAS2/group-northoff/NIMH_source_reconstruction/fsaverage_bem"
fname_fsaverage_src = pathjoin(fsaverage_bem_path, "fsaverage-ico-4-src.fif")
src = mne.read_source_spaces(fname_fsaverage_src)

subjs_common = get_commonsubj()
nsubj = len(subjs_common)

# Setup data
data_root = "/BICNAS2/group-northoff/NIMH_healthy_volunteer"
preprocpath = "/BICNAS2/group-northoff/NIMH_healthy_volunteer/preprocessing"

source_results_dir = "/BICNAS2/ycatal/erf_acw2/results/erf_source"
output_dir = pathjoin(source_results_dir, taskname)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

tasknames = [
    "encode_face_happy",
    "encode_face_sad",
    "encode_shape",
    "probe_face_happy",
    "probe_face_sad",
    "probe_shape",
]

n_vertices_morphed = 5124

tasks = {i: np.zeros((nsubj, 1201, n_vertices_morphed)) for i in tasknames}
task_stcs = {i: [] for i in tasknames}

for i, i_subj in enumerate(subjs_common):
    subj_preprocpath = pathjoin(preprocpath, i_subj)
    data_path = pathjoin(subj_preprocpath, "haririhammer", "morphing")
    for j in tasknames:
        filename_morph = pathjoin(data_path, f"{j}_morph-morph.h5")
        morph = mne.read_source_morph(filename_morph)
        filename_stc = pathjoin(subj_preprocpath, "haririhammer", f"{j}_stc.pkl")
        stc = pklload(filename_stc)
        stc_morphed = morph.apply(stc)
        task_stcs[j].append(stc_morphed)
        i_data = stc_morphed.data.T
        tasks[j][i, :, :] = i_data

    print(f"{i+1} / {nsubj}")

grand_averages = {i: np.mean(task_stcs[i]) for i in tasknames}
grand_averages_std = {i: np.nanstd(task_stcs[i]) for i in tasknames}  

fname = pathjoin(output_dir, "grand_average.pkl")
pklsave(fname, {"grand_averages":grand_averages, "grand_averages_std":grand_averages_std})
