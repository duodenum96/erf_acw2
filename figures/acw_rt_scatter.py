import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from os.path import join
import pingouin as pg

data = pd.read_csv("/BICNAS2/ycatal/erf_acw2/results/data_st.csv")
figpath = "/BICNAS2/ycatal/erf_acw2/figures/figs/figure5"
data_ave = (
    data.drop_duplicates(subset=["channels", "subjects"])
    .groupby(["subjects"])[["restacws", "rts"]]
    .mean()
    .reset_index()
)
data_ave.rename(mapper={"restacws": "ACW (s)", "rts": "RT (s)"}, inplace=True)
plt.rcParams["font.size"] = 16
f = sns.lmplot(data=data_ave, x="ACW (s)", y="RT (s)", palette="magma")
f.savefig(join(figpath, "acw_rt_scatter.png"))

