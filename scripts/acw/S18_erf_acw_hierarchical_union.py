# cd /BICNAS2/ycatal/erf_acw2/scripts/acw
# nohup python S18_erf_acw_hierarchical_union.py > log/S18_erf_acw_hierarchical_union.log &
# pid: 1966696
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

data = pd.read_csv("/BICNAS2/ycatal/erf_acw2/results/data_st.csv")

data2 = data.copy()
data2["ERF"] = zscore(data2["erfs"])
data2["ACW"] = zscore(data2["restacws"])
data2["RT"] = zscore(data2["rts"])
data3 = data2.copy()
data3["Channel"] = [str(i) for i in data2["channels"]]
data3["Trial"] = data2["erftype"]
data3 = data3.dropna()

data3["ClusterIdx"] = np.nan
data3["Cluster"] = ""


cluster_conditions = [
    data3["clusters"].isin([0, 1]),
    data3["clusters"].isin([2, 3]), 
    data3["clusters"].isin([4, 5])
]
cluster_values = ["0", "1", "2"]
cluster_names = ["Encode #0", "Encode #1", "Probe"]

data3["ClusterIdx"] = np.select(cluster_conditions, cluster_values, default="")
data3["Cluster"] = np.select(cluster_conditions, cluster_names, default="")


model_formula = (
    "ACW ~ 1 + ERF + (1|Cluster) + (1|Channel) + (1|Trial)"
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

model = bmb.Model(model_formula, data3, priors=model_priors, family="t", noncentered=True, categorical=["Channel", "Trial", "Cluster"])
model.build()

idata = model.fit(target_accept=0.9)

idata.to_netcdf(join(resultpath, "idata_acw_erf_union.nc"))

summary = az.summary(idata)
summary

bad_rhats = summary[summary["r_hat"] > 1.01]
bad_rhats  # no bad rhats

ess_bulk = summary["ess_bulk"]
ess_bulk[ess_bulk < 300]  # no bad ess

print("done")