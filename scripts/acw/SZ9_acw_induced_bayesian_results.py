# /BICNAS2/ycatal/erf_acw2/scripts/acw/SZ9_acw_induced_bayesian_results.py

import pymc as pm
import bambi as bmb
import arviz as az
import pandas as pd
import matplotlib.pyplot as plt
from os.path import join
import os
import numpy as np

from erf_acw2.src import p_direction, rope, plot_channel_effects, create_channel_mask
from erf_acw2.src_importdata import get_template_bad
from erf_acw2.src import pklload

def zscore(x):
    return (x - np.mean(x)) / np.std(x)

# Paths
figpath = "/BICNAS2/ycatal/erf_acw2/figures/figs/reviewer_comments/induced"
figpath_supp = figpath  # keep together
os.makedirs(figpath, exist_ok=True)
os.makedirs(figpath_supp, exist_ok=True)

# Reconstruct the modeling dataset (same as SZ8), to rebuild the model for PPC
nchan = 272
subjs_common = None  # lazy load via get_commonsubj() inside src_importdata/get_template_bad if needed

# ACW integrals (rest)
acw_ints = np.nanmean(
    pklload("/BICNAS2/ycatal/erf_acw2/results/int/rest_int.pkl")["rest_acw_50s"],
    axis=0,
)

# Induced power data bundle
induced_bundle = pklload("/BICNAS2/ycatal/erf_acw2/results/erf/all_data_induced_fooof.pkl")
selected_freqbands = induced_bundle["selected_freqbands"]
tasknames = induced_bundle["tasknames"]
induced_data = induced_bundle["all_data"]

# Nice names as used during fitting
freq_nicenames = ["Theta", "Beta", "Gamma"]
task_nicenames = [
    "Encode Face Happy", "Encode Face Sad", "Encode Shape",
    "Probe Face Happy", "Probe Face Sad", "Probe Shape"
]

# Flatten to long format
all_acws = []
all_induceds = []
all_freqs = []
all_tasks = []
all_chans = []
all_subjs = []

for i in range(nchan):
    for j, j_task in enumerate(tasknames):
        for k, k_freq in enumerate(selected_freqbands):
            for m in range(56):
                all_acws.append(acw_ints[i, m])
                all_induceds.append(induced_data[j_task][k_freq][m, i])
                all_freqs.append(freq_nicenames[k])
                all_tasks.append(task_nicenames[j])
                all_chans.append(i)
                all_subjs.append(m)

data = pd.DataFrame({
    "ACW": all_acws,
    "Induced": all_induceds,
    "FrequencyBand": all_freqs,
    "Trial": all_tasks,
    "Channel": all_chans,
    "Subject": all_subjs,
})

data2 = data.dropna().copy()
data3 = data2.copy()
data3["ACW"] = zscore(data3["ACW"])
data3["Induced"] = zscore(data3["Induced"])
# Keep Channel as string to avoid accidental float formatting
data3["Channel"] = [str(i) for i in data3["Channel"]]

# Build the model (same structure as SZ8) for PPC usage
model_formula = (
    "ACW ~ 1 + Induced + (1|FrequencyBand) + (1|Trial) + (1|Channel)"
    " + (Induced|FrequencyBand) + (Induced|Trial) + (Induced|Channel)"
)

model_priors = {
    "sigma": bmb.Prior("Exponential", lam=1),
    "Induced": bmb.Prior("Normal", mu=0, sigma=1),
    "Intercept": bmb.Prior("Normal", mu=0, sigma=1),
    "1|FrequencyBand": bmb.Prior("Normal", mu=0, sigma=bmb.Prior("Exponential", lam=1)),
    "1|Trial": bmb.Prior("Normal", mu=0, sigma=bmb.Prior("Exponential", lam=1)),
    "1|Channel": bmb.Prior("Normal", mu=0, sigma=bmb.Prior("Exponential", lam=1)),
    "Induced|FrequencyBand": bmb.Prior("Normal", mu=0, sigma=bmb.Prior("Exponential", lam=1)),
    "Induced|Trial": bmb.Prior("Normal", mu=0, sigma=bmb.Prior("Exponential", lam=1)),
    "Induced|Channel": bmb.Prior("Normal", mu=0, sigma=bmb.Prior("Exponential", lam=1)),
    "nu": bmb.Prior("Exponential", lam=1),
}

model = bmb.Model(
    model_formula,
    data3,
    priors=model_priors,
    family="t",
    noncentered=True,
    categorical=["Channel", "Trial", "FrequencyBand"],
)
model.build()

# Load fitted results
idata_path = join(figpath, "acw_induced_bayesian.nc")
results = az.from_netcdf(idata_path)

# Assign readable coordinates
trial_names = ["efh", "efs", "es", "pfh", "pfs", "ps"]
freq_labels = ["Theta", "Beta", "Gamma"]
unique_channels = data3["Channel"].unique()

results = results.assign_coords(
    {
        "FrequencyBand": freq_labels,
        "Trial": trial_names,
        "Channel": unique_channels,
        "FrequencyBand__factor_dim": freq_labels,
        "Trial__factor_dim": trial_names,
        "Channel__factor_dim": unique_channels,
    }
)

# Create combined main + random-slope variables
results.posterior["Induced + Induced|Channel"] = (
    results.posterior["Induced"] + results.posterior["Induced|Channel"]
)
results.posterior["Induced + Induced|Trial"] = (
    results.posterior["Induced"] + results.posterior["Induced|Trial"]
)
results.posterior["Induced + Induced|FrequencyBand"] = (
    results.posterior["Induced"] + results.posterior["Induced|FrequencyBand"]
)
results.posterior["Frequency"] = results.posterior["FrequencyBand"]
results.posterior["Induced|Frequency\n"] = results.posterior["Induced|FrequencyBand"]
results.posterior["Induced +\nInduced|Frequency\n"] = results.posterior["Induced + Induced|FrequencyBand"]

# Summary with pd and rope
summary = az.summary(results, hdi_prob=0.94)

varnames_for_pd_rope = [
    "Induced",
    "1|Channel",
    "1|Trial",
    "1|FrequencyBand",
    "Induced|Channel",
    "Induced|Trial",
    "Induced|FrequencyBand",
    "Induced + Induced|Channel",
    "Induced + Induced|Trial",
    "Induced + Induced|FrequencyBand",
]

summary["pd"] = np.nan
summary["rope"] = np.nan

coordinates_dict = {
    "Channel": unique_channels,
    "Trial": trial_names,
    "FrequencyBand": freq_labels,
}

for varname in varnames_for_pd_rope:
    pd_val = p_direction(results, varname)
    r = rope(results, varname)
    if isinstance(r, np.float64) or np.isscalar(r):
        summary.loc[varname, "pd"] = pd_val
        summary.loc[varname, "rope"] = r
    else:
        # Vector per level
        base = varname
        coord_key = None
        if "|Channel" in varname:
            coord_key = "Channel"
        elif "|Trial" in varname:
            coord_key = "Trial"
        elif "|FrequencyBand" in varname:
            coord_key = "FrequencyBand"
        if coord_key is None:
            continue
        for i in range(len(r)):
            name = f"{base}[{coordinates_dict[coord_key][i]}]"
            summary.loc[name, "pd"] = pd_val[i]
            summary.loc[name, "rope"] = r[i]

summary.to_csv(join(figpath_supp, "summary_acw_induced.csv"))

# Prior and posterior predictive checks
prior_results = model.prior_predictive()
model.predict(results, kind="response")

plt.rcParams.update({"font.size": 12})
f, ax = plt.subplots(2, 2, figsize=(10, 10), dpi=300)
az.plot_ppc(prior_results, ax=ax[0, 0], num_pp_samples=20, group="prior")
ax[0, 0].set_xlim(-15, 15)
ax[0, 0].set_title("Prior predictive check")
ax[0, 0].set_xlabel("")

az.plot_ppc(prior_results, ax=ax[0, 1], num_pp_samples=20, group="prior", kind="cumulative")
ax[0, 1].set_xlim(-15, 15)
ax[0, 1].set_title("Prior predictive check")
ax[0, 1].set_xlabel("")

az.plot_ppc(results, ax=ax[1, 0], num_pp_samples=20, group="posterior")
ax[1, 0].set_xlim(-5, 5)
ax[1, 0].set_title("Posterior predictive check")
ax[1, 0].set_xlabel("ACW")

az.plot_ppc(results, ax=ax[1, 1], num_pp_samples=20, group="posterior", kind="cumulative")
ax[1, 1].set_xlim(-5, 5)
ax[1, 1].set_title("Posterior predictive check")
ax[1, 1].set_xlabel("ACW")

f.savefig(join(figpath_supp, "prior_and_posterior_acw_induced.png"), dpi=300)
plt.close(f)

# Trace plots
var_names = [
    "Intercept",
    "Induced",
    "nu",
    "sigma",
    "1|Trial",
    "1|FrequencyBand",
    "1|Channel",
    "1|Trial_sigma",
    "1|FrequencyBand_sigma",
    "1|Channel_sigma",
    "Induced|Trial",
    "Induced|FrequencyBand",
    "Induced|Channel",
    "Induced|Trial_sigma",
    "Induced|FrequencyBand_sigma",
    "Induced|Channel_sigma",
]

plt.close()
az.plot_trace(
    results,
    combined=True,
    compact=True,
    figsize=(12, 58),
    var_names=var_names,
)
plt.savefig(join(figpath_supp, "trace_acw_induced.png"), dpi=300)
plt.close()

results.posterior["Induced +\nInduced|Channel"] = (
    results.posterior["Induced"] + results.posterior["Induced|Channel"]
)
results.posterior["Induced +\nInduced|Trial\n"] = (
    results.posterior["Induced"] + results.posterior["Induced|Trial"]
)
results.posterior["Induced +\nInduced|FrequencyBand\n"] = (
    results.posterior["Induced"] + results.posterior["Induced|FrequencyBand"]
)

# Forest plots
plt.rcParams.update({"font.size": 24})
var_names_forest = [
    "Induced", "Induced|Trial", "Induced|Frequency\n",
    "Induced +\nInduced|Trial\n", "Induced +\nInduced|Frequency\n"
]
f, ax = plt.subplots(1, 1, figsize=(32, 22), dpi=300)
az.plot_forest(
    results,
    var_names=var_names_forest,
    combined=True, ax=ax, rope=[-0.1, 0.1], colors="black"
)
ax.set_title("ACW ~ Induced + Induced|Trial + Induced|FrequencyBand + Induced|Channel")
plt.yticks(fontsize=24)
plt.xticks(fontsize=24)
ax.axvline(0, color="black", linestyle="--")
f.savefig(join(figpath_supp, "forest_acw_induced.png"), dpi=300, transparent=True)
plt.close(f)

###### Topoplot of Channel Specific Effects #######
# For channel locations, reuse the existing CSV with channel coordinates
coords_df = pd.read_csv("/BICNAS2/ycatal/erf_acw2/results/data_st.csv")
template_good = get_template_bad(good=True)
chans = coords_df.dropna().channels.unique()
chan_order = np.argsort(chans)
info = template_good.pick(chans).info
channel_coords = coords_df.groupby("channels")[["xcoords", "ycoords"]].mean()
coords = channel_coords.values

plt.rcParams.update({"font.size": 24})

# Induced|Channel
mask_induced = create_channel_mask(summary, "Induced|Channel")
data_topo = results.posterior["Induced|Channel"].mean(dim=["chain", "draw"]).values
hdi_topo = az.hdi(results, var_names="Induced|Channel", hdi_prob=0.94)["Induced|Channel"].values
vlims = (np.min(hdi_topo), np.max(hdi_topo))

plot_channel_effects(
    data=data_topo,
    hdi_data=hdi_topo,
    info=info,
    chan_order=chan_order,
    vlim=vlims,
    title="Induced|Channel",
    figpath=join(figpath, "channels_induced_acw.jpg"),
    mask=mask_induced
)

# Induced + Induced|Channel
mask_induced_plus = create_channel_mask(summary, "Induced + Induced|Channel")
data_topo = results.posterior["Induced + Induced|Channel"].mean(dim=["chain", "draw"]).values
hdi_topo = az.hdi(results, var_names="Induced + Induced|Channel", hdi_prob=0.94)["Induced + Induced|Channel"].values
vlims = (np.min(hdi_topo), np.max(hdi_topo))

plot_channel_effects(
    data=data_topo,
    hdi_data=hdi_topo,
    info=info,
    chan_order=chan_order,
    vlim=vlims,
    title="Induced + Induced|Channel",
    figpath=join(figpath, "channels_induced_acw_plus.jpg"),
    mask=mask_induced_plus
)
