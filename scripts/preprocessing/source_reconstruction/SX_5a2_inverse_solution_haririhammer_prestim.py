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
from erf_acw2.src import pklload, pklsave

subjects_dir = "/BICNAS2/group-northoff/NIMH_source_reconstruction"

i = int(sys.argv[1])
subjlist = get_commonsubj()
i_subj = subjlist[i]

if i_subj in exclude:
    sys.exit("Bad Subject")

# Setup data
data_root = "/BICNAS2/group-northoff/NIMH_healthy_volunteer"
subj_raw_path = pathjoin(data_root, i_subj, "ses-01", "meg")

preprocpath = "/BICNAS2/group-northoff/NIMH_healthy_volunteer/preprocessing"

subj_preprocpath = pathjoin(preprocpath, i_subj)
outputpath = pathjoin(subj_preprocpath, "haririhammer", "rejection_ica")

filename = pathjoin(
    "/BICNAS2/ycatal/erf_acw2/results/erf/haririhammer",
    f"{i_subj}_haririhammer_erf_emo.pkl"
)

data = pklload(filename)

encode_face_happy = data["encode_face_happy"]
probe_face_happy = data["probe_face_happy"]
encode_face_sad = data["encode_face_sad"]
probe_face_sad = data["probe_face_sad"]
probe_shape = data["probe_shape"]
encode_shape = data["encode_shape"]

data_evokeds = [encode_face_happy, probe_face_happy, encode_face_sad, probe_face_sad, probe_shape, encode_shape]
tasknames = ["encode_face_happy", "probe_face_happy", "encode_face_sad", "probe_face_sad", "probe_shape", "encode_shape"]

# See https://mne.discourse.group/t/estimating-the-regularization-parameter-for-hcp-meg-resting-state-source-reconstruction/5177/15
snr = 3.0
lambda2 = 1.0 / snr**2

# Load the required ingredients: covariance matrix, forward solution, source space
fwd = mne.read_forward_solution(pathjoin(subj_preprocpath, "haririhammer", "forward-fwd.fif"))
src = fwd["src"]

# Load noise covariance
noise_cov = mne.read_cov(pathjoin(subj_preprocpath, "noise", "rejection_ica", f"{i_subj}_prestim-cov.fif"))

# Compute inverse solution
method = "dSPM"  # could choose MNE, sLORETA, or eLORETA instead
for data_evoked, taskname in zip(data_evokeds, tasknames):
    inverse_operator = mne.minimum_norm.make_inverse_operator(
        data_evoked.info, fwd, noise_cov
    )

    # mne.minimum_norm.write_inverse_operator(
    #     pathjoin(subj_preprocpath, "haririhammer", "inverse-inv.fif"), inverse_operator
    # )

    stc = mne.minimum_norm.apply_inverse(
        data_evoked,
        inverse_operator,
        lambda2,
        method=method,
        verbose=True,
    )

    pklsave(pathjoin(subj_preprocpath, "haririhammer", f"{taskname}_stc.pkl"), stc)

    # vertno_max, time_max = stc.get_peak(hemi="rh")

    # surfer_kwargs = dict(
    #     hemi="rh",
    #     subjects_dir=subjects_dir,
    #     clim=dict(kind="value", lims=[8, 12, 15]),
    #     views="lateral",
    #     initial_time=time_max,
    #     time_unit="s",
    #     size=(800, 800),
    #     smoothing_steps=10,
    # )
    # brain = stc.plot(**surfer_kwargs)
    # brain.add_foci(
    #     vertno_max,
    #     coords_as_verts=True,
    #     hemi="rh",
    #     color="blue",
    #     scale_factor=0.6,
    #     alpha=0.5,
    # )
    # brain.add_text(
    #     0.1, 0.9, "dSPM (plus location of maximal activation)", "title", font_size=14
    # )

print("Done")
