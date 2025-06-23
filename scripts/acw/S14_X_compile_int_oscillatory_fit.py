import numpy as np
from erf_acw2.src import pklload, pklsave, get_commonsubj
from erf_acw2.meg_chlist import chlist

chlist = np.asarray(chlist)
import matplotlib.pyplot as plt
import matplotlib.cm as cm

nchan = 272

task = "haririhammer"

rest_ntp = 120
rest_ntp = int((rest_ntp * 3) / 10)

subjs_common = get_commonsubj()
nsubj = subjs_common.shape[0]

nlags = 300
rest_acws = np.zeros((rest_ntp, nchan, nsubj))
popts = np.zeros((rest_ntp, nchan, nsubj, 3))

all_chidx = np.arange(272)
######## Load rest and task ACWs, store in a numpy array
for i, i_subj in enumerate(subjs_common):
    restname = f"/BICNAS2/ycatal/erf_acw2/results/int/rest/{i_subj}_rest_acw_oscillatory_fit.pkl"

    rest_i_acw = pklload(restname)

    # If there is a missing channel, find it and fill with nans
    missingchan_rest = np.setdiff1d(chlist, rest_i_acw["chanlist"])
    if len(missingchan_rest) != 0:
        missing_idx_rest = np.where(chlist == missingchan_rest)[0]
        good_idx_rest = np.setdiff1d(all_chidx, missing_idx_rest)

        rest_acws[:, missing_idx_rest, i] = np.nan
        popts[:, missing_idx_rest, i, :] = np.nan
    else:
        good_idx_rest = all_chidx.copy()

    rest_acws[:, good_idx_rest, i] = rest_i_acw["acws"]
    popts[:, good_idx_rest, i, :] = rest_i_acw["popts"]

allacws = {"rest_acws": rest_acws, "popts": popts}

savename = f"/BICNAS2/ycatal/erf_acw2/results/int/rest_int_oscillatory_fit.pkl"
pklsave(savename, allacws)

############################# 
import pingouin as pg
from scipy import stats

all_acw50s = pklload(f"/BICNAS2/ycatal/erf_acw2/results/int/rest_int.pkl")["rest_acw_50s"]
mean_all_acw50s = np.nanmean(all_acw50s, axis=0)

corr_result = pg.corr(all_acw50s.ravel(), rest_acws.ravel(), method="spearman")

f, ax = plt.subplots()

ax.scatter(all_acw50s.ravel(), rest_acws.ravel(), color="black", s=1)
ax.set_xlabel("ACW-50")
ax.set_ylabel("$ \\tau $")

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.text(0.01, 0.3, f"r = {corr_result['r'].values[0]:.3f}, p = {corr_result['p-val'].values[0]:.3f}", ha="left", va="top")

# Remove NaN values before linear regression
mask = ~(np.isnan(all_acw50s.ravel()) | np.isnan(rest_acws.ravel()))
slope, intercept, r_value, p_value, std_err = stats.linregress(all_acw50s.ravel()[mask], rest_acws.ravel()[mask])

# Plot regression line
x_line = np.array([all_acw50s.ravel()[mask].min(), all_acw50s.ravel()[mask].max()])
y_line = slope * x_line + intercept
ax.plot(x_line, y_line, 'r--', alpha=0.8, linewidth=2)
        
f.savefig(f"/BICNAS2/ycatal/erf_acw2/figures/figs/figure5/acw_50s_tau.png")
