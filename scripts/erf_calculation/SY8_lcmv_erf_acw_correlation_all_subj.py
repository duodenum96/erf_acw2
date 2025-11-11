import numpy as np
import os
from os.path import join as pathjoin
import pingouin as pg

os.chdir("/BICNAS2/ycatal/erf_acw2/")

from erf_acw2.src import (
    pklload,
    pklsave,
    get_commonsubj
)

def zscore(x):
    return (x - np.mean(x)) / np.std(x)

taskname = "haririhammer"
tasknames = [
    "encode_face_happy",
    "probe_face_happy",
    "encode_face_sad",
    "probe_face_sad",
    "probe_shape",
    "encode_shape",
]

source_results_dir = "/BICNAS2/ycatal/erf_acw2/results/erf_source"
source_output_dir = pathjoin(source_results_dir, taskname)

significant_data = pklload(pathjoin(source_output_dir, "circular_shift_stats.pkl"))
all_erfs = pklload(pathjoin(source_output_dir, "all_data.pkl"))
erf_data = all_erfs["all_data"]
task_times = all_erfs["times"]
poststim_times = np.where((task_times > 0.) & (task_times < 0.6))[0]
poststim_erfs = erf_data[:, :, :, poststim_times]

int_results_dir = "/BICNAS2/ycatal/erf_acw2/results/int"
int_data = pklload(pathjoin(int_results_dir, "source_lcmv_acw_oscillatory_fit.pkl"))["popt_all"]
ints = int_data[:, :, 1]
# ints = pklload(pathjoin(int_results_dir, "source_lcmv_acw_oscillatory_fit.pkl"))["acws_all"]

sig_indices = significant_data["significant_indices"] 
##################################################################################################
import pandas as pd
import seaborn as sns
import arviz as az

nsubj = ints.shape[0]

ctx_indices = np.array(["ctx" in i for i in labels])
ctx_labels = labels[ctx_indices]

ctx_ints = ints[:, ctx_indices]
ctx_erfs = poststim_erfs[:, :, ctx_indices, :]

sig_ints = ctx_ints[sig_indices[0], sig_indices[2]]
significant_stuff = ctx_erfs[sig_indices[0], sig_indices[1], sig_indices[2], :]
poststim_average = np.sqrt(significant_stuff**2).mean(axis=1)

pg.corr(sig_ints, poststim_average, method="spearman")

roi_ints = {i: {j: [] for j in tasknames} for i in ctx_labels}
roi_erfs = {i: {j: [] for j in tasknames} for i in ctx_labels}
roi_subjs = {i: {j: [] for j in tasknames} for i in ctx_labels}
for i, i_roi in enumerate(sig_indices[2]):
    for k in range(nsubj):
        roi_ints[ctx_labels[i_roi]][tasknames[sig_indices[1][i]]].append(ctx_ints[k, i_roi])
        roi_erfs[ctx_labels[i_roi]][tasknames[sig_indices[1][i]]].append(
            np.mean(
                np.sqrt(ctx_erfs[k, sig_indices[1][i], i_roi, :] ** 2)
            )
        )
        roi_subjs[ctx_labels[i_roi]][tasknames[sig_indices[1][i]]].append(k)


all_ints = []
all_erfs = []
all_labels = []
all_tasknames = []
all_subj = []
for i_label in ctx_labels:
    for j in tasknames:
        n_measures = len(roi_ints[i_label][j])
        for k in range(n_measures):
            all_ints.append(roi_ints[i_label][j][k])
            all_erfs.append(roi_erfs[i_label][j][k])
            all_labels.append(i_label)
            all_tasknames.append(j)
            all_subj.append(roi_subjs[i_label][j][k])
        
df = pd.DataFrame({"ints": all_ints, "erfs": all_erfs, "labels": all_labels, "tasknames": all_tasknames,
                   "subjs": all_subj})

df["ints_z"] = zscore(df["ints"])
df["erfs_z"] = zscore(df["erfs"])

import bambi as bmb

model = bmb.Model("ints_z ~ erfs_z + (erfs_z | labels) + (erfs_z | tasknames)", df.dropna(), family="t")
results = model.fit()
summary = az.summary(results)
summary.to_csv("anan_allsubj.csv")

# sns.FacetGrid(data=df, row="labels", col="tasknames").map(sns.scatterplot, "ints", "erfs")
# plt.savefig("anan.jpg")

import matplotlib.pyplot as plt
plt.close()
plt.scatter(sig_ints, poststim_average, s=1, c="k")
# plt.xlim((0, 0.03))
plt.savefig("anan_subjthreshold.jpg")

df_grouped = df.groupby(['subjs', 'tasknames']).agg({
    'ints': 'mean',
    'erfs': 'mean'
}).reset_index()
df_grouped[["ints", "erfs"]].corr()
df_grouped[df_grouped["tasknames"] == "probe_shape"][["ints", "erfs"]].corr()
##################################################################################################

# sig_indices: list of length 3
# first element: subject indices
# second element: task indices
# third element: roi indices

for i, i_task in enumerate(tasknames):
    sig_rois = np.unique(sig_indices[2][sig_indices[1] == i])
    sig_subjects = np.unique(sig_indices[0][sig_indices[1] == i])
    # sig_ints = ints[sig_subjects, :][:, sig_rois]
    sig_ints = ints[:, sig_rois]
    # sig_erfs = np.nanmean(poststim_erfs[sig_subjects, :, :, :][:, i, :, :][:, sig_rois, :], axis=-1)
    sig_erfs = np.nanmean(np.sqrt(poststim_erfs[:, i, sig_rois, :]**2), axis=-1)
    corr_results = pg.corr(sig_ints.ravel(), sig_erfs.ravel())
    print(f"Task {i_task}: {corr_results['r'].values[0]:.3f}, p = {corr_results['p-val'].values[0]:.3f}")


# in general
all_corr_results = []

for i in range(185):
    for j in range(len(tasknames)):
        result = pg.corr(ints[:, i], np.nanmean(np.sqrt(poststim_erfs[:, j, i, :]**2), axis=-1))
        all_corr_results.append(result["r"].values[0])

np.mean(np.array(all_corr_results) > 0)
np.mean(np.array(all_corr_results))

pg.corr(np.array(all_corr_results), np.nanmean(
    np.nanmean(np.sqrt(poststim_erfs**2), axis=-1), axis=0).ravel()
    )