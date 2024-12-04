import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import bambi as bmb
import pymc as pm
import arviz as az
import os
from os.path import join
import cloudpickle
import pingouin as pg
import statsmodels.api as sm

def zscore(x):
    return (x - x.mean()) / x.std()


def codify(df, varname):
    uniques = df[varname].unique()
    mapping = {ch: i for i, ch in enumerate(uniques)}
    code = df[varname].map(mapping)
    return code

rootpath = r"C:\Users\duodenum\Desktop\brain_stuff\erf_acw_bayes"
os.chdir(rootpath)

figpath = join(rootpath, "figure5")

def cloud_pklsave(filename, obj):
    with open(filename, "wb") as f:
        cloudpickle.dump(obj, f)


data = pd.read_csv("data_st.csv")
data = data.rename(
    columns={
        "erfs": "ERF",
        "restacws": "ACW",
        "rts": "RT",
        "clusters": "Cluster",
        "subjects": "Subject",
        "channels": "Channel",
        "erftype": "Trial",
    }
)

data2 = data.copy()
data2["ERF"] = zscore(data2["ERF"])
data2["ACW"] = zscore(data2["ACW"])
data2["RT"] = zscore(data2["RT"])
data3 = data2.copy()
data3["Channel"] = [str(i) for i in data2["Channel"]]
data3 = data3.dropna()

model_formula = (
    "RT ~ 1 + ERF + (1|Cluster) + (1|Channel) + (1|Trial)"
    " + (ERF|Cluster) + (ERF|Channel) + (ERF|Trial)"
)
model_priors = {
    "sigma": bmb.Prior("Exponential", lam=1),
    "ERF": bmb.Prior("Normal", mu=0, sigma=1),
    "Intercept": bmb.Prior("Normal", mu=0, sigma=1),
    "1|Channel": bmb.Prior("Normal", mu=0, sigma=bmb.Prior("Exponential", lam=1)),
    "1|Trial": bmb.Prior("Normal", mu=0, sigma=bmb.Prior("Exponential", lam=1)),
    "1|Cluster": bmb.Prior("Normal", mu=0, sigma=bmb.Prior("Exponential", lam=1)),
    "ERF|Channel": bmb.Prior("Normal", mu=0, sigma=bmb.Prior("Exponential", lam=1)),
    "ERF|Trial": bmb.Prior("Normal", mu=0, sigma=bmb.Prior("Exponential", lam=1)),
    "ERF|Cluster": bmb.Prior("Normal", mu=0, sigma=bmb.Prior("Exponential", lam=1)),
    "nu": bmb.Prior("Exponential", lam=1),
}

model = bmb.Model(
    model_formula,
    data3,
    priors=model_priors,
    family="t",
    noncentered=True,
    categorical=["Cluster", "Channel", "Trial"],
)
model.build()

idata = model.fit(target_accept=0.99, tune=4000)  # no divergences

idata.to_netcdf("idata_rt_erf.nc")

print("done")

summary = az.summary(idata)
summary

bad_rhats = summary[summary["r_hat"] > 1.01]
bad_rhats  # no bad rhats

ess_bulk = summary["ess_bulk"]
ess_bulk[ess_bulk < 300]  # no bad ess
