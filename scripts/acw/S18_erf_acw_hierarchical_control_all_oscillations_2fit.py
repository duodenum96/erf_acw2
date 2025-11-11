# cd /BICNAS2/ycatal/erf_acw2/scripts/acw
# nohup python S18_erf_acw_hierarchical_control_all_oscillations_2fit.py > log/hierarchical_nuts_erf_acw_control_all_oscillations_2fit.log &
# echo $! > log/pid_hierarchical_nuts_erf_acw_control_all_oscillations_2fit.txt
import os
import numpy as np
import pandas as pd
import pymc as pm
from scipy.spatial.distance import pdist, squareform
import cloudpickle
from os.path import join
import arviz as az
import bambi as bmb

def zscore(x):
    return (x - x.mean()) / x.std()


def codify(df, varname):
    uniques = df[varname].unique()
    mapping = {ch: i for i, ch in enumerate(uniques)}
    code = df[varname].map(mapping)
    return code


np.random.default_rng(666)
os.chdir("/BICNAS2/ycatal/erf_acw2/")

figpath = "/BICNAS2/ycatal/erf_acw2/figures/figs/figure5"
resultpath = "/BICNAS2/ycatal/erf_acw2/results/hierarchical_model"
def cloud_pklsave(filename, obj):
    with open(filename, "wb") as f:
        cloudpickle.dump(obj, f)

data = pd.read_csv("/BICNAS2/ycatal/erf_acw2/results/data_st_and_oscillations_2fit.csv")

data2 = data.copy()
data2["ERF"] = zscore(data2["erfs"])
data2["ACW"] = zscore(data2["restacws"])
data2["RT"] = zscore(data2["rts"])
data2["Alpha"] = zscore(data2["alphas"])
data2["Beta1"] = zscore(data2["beta1s"])
data2["Beta2"] = zscore(data2["beta2s"])
data2["Gamma"] = zscore(data2["gammas"])
data2["Theta"] = zscore(data2["thetas"])
data3 = data2.copy()
data3["Channel"] = [str(i) for i in data2["channels"]]
data3 = data3.dropna()
data3["Cluster"] = data2.clusternames
data3["Trial"] = data2.erftype


model_formula = (
    "ACW ~ 1 + ERF + Alpha + Beta1 + Beta2 + Gamma + Theta + (1|Cluster) + (1|Channel) + (1|Trial)"
    " + (ERF|Cluster) + (ERF|Channel) + (ERF|Trial) + (Alpha|Channel) + (Beta1|Channel) + (Beta2|Channel) + (Gamma|Channel) + (Theta|Channel)"
)

model_priors = {
    "sigma": bmb.Prior("Exponential", lam=1),
    "ERF": bmb.Prior("Normal", mu=0, sigma=1),
    "Alpha": bmb.Prior("Normal", mu=0, sigma=1),
    "Beta1": bmb.Prior("Normal", mu=0, sigma=1),
    "Beta2": bmb.Prior("Normal", mu=0, sigma=1),
    "Gamma": bmb.Prior("Normal", mu=0, sigma=1),
    "Theta": bmb.Prior("Normal", mu=0, sigma=1),
    "Intercept": bmb.Prior("Normal", mu=0, sigma=1),
    "1|Channel": bmb.Prior("Normal", mu=0, sigma=bmb.Prior("Exponential", lam=1)),
    "1|Trial": bmb.Prior("Normal", mu=0, sigma=bmb.Prior("Exponential", lam=1)),
    "1|Cluster": bmb.Prior("Normal", mu=0, sigma=bmb.Prior("Exponential", lam=1)),
    "ERF|Channel": bmb.Prior("Normal", mu=0, sigma=bmb.Prior("Exponential", lam=1)),
    "ERF|Trial": bmb.Prior("Normal", mu=0, sigma=bmb.Prior("Exponential", lam=1)),
    "ERF|Cluster": bmb.Prior("Normal", mu=0, sigma=bmb.Prior("Exponential", lam=1)),
    "Alpha|Channel": bmb.Prior("Normal", mu=0, sigma=bmb.Prior("Exponential", lam=1)),
    "Beta1|Channel": bmb.Prior("Normal", mu=0, sigma=bmb.Prior("Exponential", lam=1)),
    "Beta2|Channel": bmb.Prior("Normal", mu=0, sigma=bmb.Prior("Exponential", lam=1)),
    "Gamma|Channel": bmb.Prior("Normal", mu=0, sigma=bmb.Prior("Exponential", lam=1)),
    "Theta|Channel": bmb.Prior("Normal", mu=0, sigma=bmb.Prior("Exponential", lam=1)),
    "nu": bmb.Prior("Exponential", lam=1),
}

model = bmb.Model(model_formula, data3, priors=model_priors, family="t", noncentered=True, categorical=["Channel", "Trial", "Cluster"])
model.build()


idata = model.fit(target_accept=0.99, tune=4000)

idata.to_netcdf(join(resultpath, "idata_acw_erf_control_all_oscillations_2fit.nc"))

summary = az.summary(idata)
summary

bad_rhats = summary[summary["r_hat"] > 1.01]
bad_rhats  # no bad rhats

ess_bulk = summary["ess_bulk"]
ess_bulk[ess_bulk < 300]  # no bad ess

print("done")