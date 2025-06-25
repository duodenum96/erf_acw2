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
fsaverage_bem_path = "/BICNAS2/group-northoff/NIMH_source_reconstruction/fsaverage_bem"


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
    f"{i_subj}_haririhammer_erf_emo.pkl",
)

inputpath = pathjoin(subj_preprocpath, "haririhammer", "rejection_ica")
epochs_filename = pathjoin(
    inputpath, i_subj + "_haririhammer_preprocessed_erf_emo-epo.fif"
)

epochs_ar2 = mne.read_epochs(epochs_filename)


data = pklload(filename)

encode_face_happy = data["encode_face_happy"]
probe_face_happy = data["probe_face_happy"]
encode_face_sad = data["encode_face_sad"]
probe_face_sad = data["probe_face_sad"]
probe_shape = data["probe_shape"]
encode_shape = data["encode_shape"]

data_evokeds = [
    encode_face_happy,
    probe_face_happy,
    encode_face_sad,
    probe_face_sad,
    probe_shape,
    encode_shape,
]
tasknames = [
    "encode_face_happy",
    "probe_face_happy",
    "encode_face_sad",
    "probe_face_sad",
    "probe_shape",
    "encode_shape",
]


# Load the required ingredients: covariance matrix, forward solution, source space
fwd = mne.read_forward_solution(
    pathjoin(subj_preprocpath, "haririhammer", "forward_volume-fwd.fif")
)
src = fwd["src"]

# Load noise covariance
noise_cov = mne.read_cov(
    pathjoin(subj_preprocpath, "noise", "rejection_ica", f"{i_subj}_prestim-cov.fif")
)

rank = mne.compute_rank(noise_cov, info=epochs_ar2.info)

labels_path = pathjoin(subjects_dir, i_subj, "mri", "aparc.a2009s+aseg.mgz")
labels = mne.get_volume_labels_from_aseg(labels_path)

# Compute inverse solution
for data_evoked, taskname in zip(data_evokeds, tasknames):
    data_cov = mne.compute_covariance(
        epochs_ar2[taskname], tmin=0.15, tmax=0.3, method="auto"
    )

    filters = mne.beamformer.make_lcmv(
        data_evoked.info,
        fwd,
        data_cov,
        reg=0.05,
        noise_cov=noise_cov,
        pick_ori="max-power",
        weight_norm="unit-noise-gain",
        rank=rank,
    )

    stc = mne.beamformer.apply_lcmv(
        data_evoked,
        filters,
    )

    parcellated_ts = mne.extract_label_time_course(
        stc,
        labels_path,
        src,  # allow_empty=True
    )

    pklsave(
        pathjoin(subj_preprocpath, "haririhammer", f"{taskname}_lcmv_stc.pkl"),
        {"stc": stc, "parcellated_ts": parcellated_ts, "labels": labels},
    )


print("Done")
