# cd /BICNAS2/ycatal/erf_acw2/scripts/acw
# nohup python SZ8_acw_induced_bayesian.py > log/SZ8_acw_induced_bayesian.log 2>&1 &
# echo $! > SZ8_acw_induced_bayesian.pid

import bambi as bmb
import pymc as pm
import numpy as np
import pandas as pd
from erf_acw2.src import get_commonsubj, pklload, pklsave
import os
import arviz as az

def z_score(x):
    return (x - np.mean(x)) / np.std(x)

figfolder = "/BICNAS2/ycatal/erf_acw2/figures/figs/reviewer_comments/induced"

nchan = 272

task = "haririhammer"

subjs_common = get_commonsubj()
nsubj = subjs_common.shape[0]

acw_ints = np.nanmean(
    pklload(f"/BICNAS2/ycatal/erf_acw2/results/int/rest_int.pkl")["rest_acw_50s"],
    axis=0
)

filename = (
    f"/BICNAS2/ycatal/erf_acw2/results/erf/" + 
    f"all_data_induced_fooof.pkl"
)

induced_data = pklload(filename)
selected_freqbands = induced_data["selected_freqbands"]
tasknames = induced_data["tasknames"]
induced_data = induced_data["all_data"]
n_tasks = len(tasknames)
n_freqs = len(selected_freqbands)

freq_nicenames = ["Theta", "Beta", "Gamma"]
task_nicenames = ["Encode Face Happy", "Encode Face Sad", "Encode Shape", 
                  "Probe Face Happy", "Probe Face Sad", "Probe Shape"]


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

data2 = data.dropna()
data3 = data2.copy()
data3["ACW"] = z_score(data3["ACW"])
data3["Induced"] = z_score(data3["Induced"])

formula = (
    "ACW ~ 1 + Induced + (1|FrequencyBand) + (1|Trial) + (1|Channel) "
    "+ (Induced|FrequencyBand) + (Induced|Trial) + (Induced|Channel)"
)
priors = {
    "sigma": bmb.Prior("Exponential", lam=1),
    "Induced": bmb.Prior("Normal", mu=0, sigma=1),
    "1|FrequencyBand": bmb.Prior("Normal", mu=0, sigma=bmb.Prior("Exponential", lam=1)),
    "1|Trial": bmb.Prior("Normal", mu=0, sigma=bmb.Prior("Exponential", lam=1)),
    "1|Channel": bmb.Prior("Normal", mu=0, sigma=bmb.Prior("Exponential", lam=1)),
    "Induced|FrequencyBand": bmb.Prior("Normal", mu=0, sigma=bmb.Prior("Exponential", lam=1)),
    "Induced|Trial": bmb.Prior("Normal", mu=0, sigma=bmb.Prior("Exponential", lam=1)),
    "Induced|Channel": bmb.Prior("Normal", mu=0, sigma=bmb.Prior("Exponential", lam=1)),
    "nu": bmb.Prior("Exponential", lam=1),
}

model = bmb.Model(formula, data3, family="t", noncentered=True, categorical=["FrequencyBand", "Trial", "Channel"], priors=priors)
model.build()

idata = model.fit(target_accept=0.99, tune=4000)

idata.to_netcdf(os.path.join(figfolder, "acw_induced_bayesian.nc"))