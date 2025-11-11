import h5py
import seaborn as sns
import pandas as pd
import numpy as np
from seaborn import objects as so
from seaborn import axes_style
import matplotlib.pyplot as plt
import pingouin as pg
from os.path import join
import matplotlib as mpl
import os
from erf_acw2.src import (
    get_commonsubj,
    pklsave,
    pklload,
    import_exampleraw,
    acf_oscillatory_function,
)

figpath = "/BICNAS2/ycatal/erf_acw2/figures/figs/model_reviewer_comments"

data = pd.read_csv(os.path.join(figpath, "erf_acw_data.csv"))

data_filtered = data[(data["erfs"].notna()) & (data["acw50s"].notna())]

data_filtered = data_filtered[data_filtered["acw50s"] < 0.05]

f = sns.FacetGrid(data_filtered, col_wrap=5, col="roi_idx", sharex=False, sharey=False).map(
    sns.regplot, "gamma_1", "acw50s", ci=None
)
f.savefig(os.path.join(figpath, "f6_erf_acw_correlation.png"))

