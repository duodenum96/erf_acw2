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

adj = mne.spatial_src_adjacency(src)

subjs_common = get_commonsubj()
nsubj = len(subjs_common)

# Setup data
data_root = "/BICNAS2/group-northoff/NIMH_healthy_volunteer"
preprocpath = "/BICNAS2/group-northoff/NIMH_healthy_volunteer/preprocessing"

source_results_dir = "/BICNAS2/ycatal/erf_acw2/results/erf_source"
output_dir = pathjoin(source_results_dir, taskname)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# MNE ANOVA f'n requires the first factor to be slow and second to be fast
# first factor: emotion (happy, sad, shape); second factor: encode vs probe
# So encode: happy, sad, shape; probe: happy, sad, shape
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

for i, i_subj in enumerate(subjs_common):
    subj_preprocpath = pathjoin(preprocpath, i_subj)
    data_path = pathjoin(subj_preprocpath, "haririhammer", "morphing")
    for j in tasknames:
        filename_morph = pathjoin(data_path, f"{j}_morph-morph.h5")
        morph = mne.read_source_morph(filename_morph)
        filename_stc = pathjoin(subj_preprocpath, "haririhammer", f"{j}_stc.pkl")
        stc = pklload(filename_stc)
        stc_morphed = morph.apply(stc)
        i_data = stc_morphed.data.T
        tasks[j][i, :, :] = i_data

    print(f"{i+1} / {nsubj}")

X = [tasks[i] for i in tasknames]
factor_levels = [2, 3]

return_pvals = False
pthresh = 0.0001
n_replications = nsubj
all_effects = ["A", "B"]
effect_names = ["factor_encprob", "factor_emo"]


def stat_fun(*args):
    # get f-values only.
    return mne.stats.f_mway_rm(
        np.swapaxes(args, 1, 0),
        factor_levels=factor_levels,
        effects=effects,
        return_pvals=return_pvals,
    )[0]


for i, effects in enumerate(all_effects):
    f_thresh = mne.stats.f_threshold_mway_rm(
        n_replications, factor_levels, effects, pthresh
    )

    stats, clusters, cluster_p, h0 = mne.stats.spatio_temporal_cluster_test(
        X,
        adjacency=adj,
        stat_fun=stat_fun,
        threshold=f_thresh,
        n_permutations=200,
        n_jobs=50,
    )

    savename = pathjoin(output_dir, f"{taskname}_erp_permutationtest_st_{effect_names[i]}.pkl")
    pklsave(
        savename,
        {"stats": stats, "clusters": clusters, "cluster_p": cluster_p, "h0": h0},
    )


