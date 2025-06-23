import mne
import numpy as np
import os
from os.path import join as pathjoin
from erf_acw2.src import pklload, pklsave
import matplotlib.pyplot as plt
import matplotlib
import pingouin as pg
import pandas as pd
from erf_acw2.src import get_commonsubj

all_ts_data = pklload(
    "/BICNAS2/ycatal/erf_acw2/results/erf_source/haririhammer/haririhammer_stc.pkl"
)  # condition: subj x time x vertices

conditions = list(all_ts_data.keys())
taskname = "haririhammer"
source_results_dir = "/BICNAS2/ycatal/erf_acw2/results/erf_source"
output_dir = pathjoin(source_results_dir, taskname)

results_dir = "/BICNAS2/ycatal/erf_acw2/results/erf_source/permutation_test"

subjs_common = get_commonsubj()

results_path = "/BICNAS2/ycatal/erf_acw2/results/source"


# Load significant clusters
fname = pathjoin(output_dir, f"{taskname}_erp_permutationtest_st_all_cluster_vis.pkl")
cluster_data = pklload(fname)
cluster_results = cluster_data["cluster_results"]
clus = cluster_data["clus"]

significant_clusters_to_save = pklload(
    os.path.join(results_dir, f"{taskname}_significant_clusters_to_save.pkl")
)

significant_clusters = significant_clusters_to_save["significant_clusters"]
vertices_in_all_clusters = significant_clusters_to_save["vertices_in_all_clusters"]

erfs = []
subjs = []
factors = []
cluster_idx = []
condition = []
vertices = []

for i_significant_cluster in significant_clusters:
    comparison = i_significant_cluster["i_comparison"]

    i_times = i_significant_cluster["i_times"]
    i_vertices = i_significant_cluster["i_vertices"]

    for i_comparison in comparison:
        i_erfs = np.mean(
            all_ts_data[i_comparison][:, np.unique(i_times), :], axis=1
        )  # subj x vertices

        for i_subj, subj in enumerate(subjs_common):
            for j, j_vertex in enumerate(np.unique(i_vertices)):
                condition.append(i_comparison)

                erfs.append(i_erfs[i_subj, j])
                subjs.append(subj)
                vertices.append(j_vertex)

                cluster_idx.append(i_significant_cluster["cluster_idx"])
                factors.append(i_significant_cluster["i_factor"])


df = pd.DataFrame(
    {
        "condition": condition,
        "subjs": subjs,
        "vertices": vertices,
        "erfs": erfs,
        "cluster_idx": cluster_idx,
        "factors": factors,
    }
)

df["factor_cluster"] = df["factors"] + "_" + df["cluster_idx"].astype(str)

df.to_csv(pathjoin(results_path, f"{taskname}_significant_erfs.csv"), index=False)

##################################### Merge with FOOOF results #####################################
# Make sure vertices match

fooof_results = pd.read_csv(pathjoin(results_path, f"acw_FOOOF_source.csv"))

df2 = pd.merge(df, fooof_results, on=["vertices", "subjs"], how="left")

# df3 = df2[ (df2["ints"] > 1e-3) & (df2["ints"] < 1) ]
df3 = df2[ (df2["ints"] > 0) ]

corr_results_r = []
corr_results_p = []
for i_factor_cluster in df3["factor_cluster"].unique():
    i_df = df3[df3["factor_cluster"] == i_factor_cluster]
    # Average ERFs over vertices for each subject and condition
    i_df_avg = i_df.groupby(["subjs", "condition"])[["erfs", "ints"]].mean().reset_index()
    for j_condition in i_df_avg["condition"].unique():
        i_df_avg_condition = i_df_avg[i_df_avg["condition"] == j_condition]
        print(j_condition)
        corr_results_r.append(pg.corr(i_df_avg_condition["erfs"], i_df_avg_condition["ints"], method="spearman")["r"])
        corr_results_p.append(pg.corr(i_df_avg_condition["erfs"], i_df_avg_condition["ints"], method="spearman")["p-val"])

corr_results_r
corr_results_p
