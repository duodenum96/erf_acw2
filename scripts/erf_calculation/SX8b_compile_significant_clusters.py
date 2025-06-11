from os.path import join as pathjoin
from erf_acw2.src import pklsave, pklload, get_commonsubj, import_exampleraw
import numpy as np
import mne

raw = import_exampleraw()

sfreq = raw.info["sfreq"]

fsaverage_bem_path = "/BICNAS2/group-northoff/NIMH_source_reconstruction/fsaverage_bem"
fname_fsaverage_src = pathjoin(fsaverage_bem_path, "fsaverage-ico-4-src.fif")
src = mne.read_source_spaces(fname_fsaverage_src)

fsave_vertices = [s["vertno"] for s in src]

taskname = "haririhammer"

all_effects = ["A", "B"]
effect_names = ["factor_encprob", "factor_emo"]

source_results_dir = "/BICNAS2/ycatal/erf_acw2/results/erf_source"

cluster_results = {}
clus = {}
output_dir = pathjoin(source_results_dir, taskname)

tstep = 1 / sfreq

for effect_name in effect_names:
    loadname = pathjoin(
        output_dir, f"{taskname}_erp_permutationtest_st_{effect_name}.pkl"
    )
    cluster_results[effect_name] = pklload(loadname)

    # good_clusters_idx = np.where(cluster_results[effect_name]["cluster_p"] < 0.05)[0]
    # good_clusters = [cluster_results[effect_name]["clusters"][idx] for idx in good_clusters_idx]
    # good_cluster_p = [cluster_results[effect_name]["cluster_p"][idx] for idx in good_clusters_idx]

    # clus[effect_name] = (cluster_results[effect_name]["stats"],
    #                      good_clusters,
    #                      good_cluster_p,
    #                      cluster_results[effect_name]["h0"])

    clus[effect_name] = (
        cluster_results[effect_name]["stats"],
        cluster_results[effect_name]["clusters"],
        cluster_results[effect_name]["cluster_p"],
        cluster_results[effect_name]["h0"],
    )

    stc_all_cluster_vis = mne.stats.summarize_clusters_stc(
        clus[effect_name],
        tstep=tstep,
        vertices=fsave_vertices,
        subject="fsaverage",
        p_thresh=0.05,
        tmin=-0.3,
    )

    stc_all_cluster_vis.save(
        pathjoin(output_dir, f"{taskname}_erp_permutationtest_st_{effect_name}_all_cluster_vis.stc"), overwrite=True
    )
    