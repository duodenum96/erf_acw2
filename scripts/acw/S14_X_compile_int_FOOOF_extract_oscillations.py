import numpy as np
from erf_acw2.src import pklload, pklsave, get_commonsubj
from erf_acw2.meg_chlist import chlist
import fooof

chlist = np.asarray(chlist)
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from fooof.bands import Bands

bands = Bands({'theta' : [4, 8],
               'alpha' : [8, 12],
               'beta' : [15, 30],
               "gamma" : [30, 50]})

nchan = 272

task = "haririhammer"

rest_ntp = 120
rest_ntp = int((rest_ntp * 3) / 10)

subjs_common = get_commonsubj()
nsubj = subjs_common.shape[0]

band_powers = {"theta": np.zeros((nchan, nsubj)),
               "beta": np.zeros((nchan, nsubj)),
               "alpha": np.zeros((nchan, nsubj)),
               "gamma": np.zeros((nchan, nsubj))}

band_fooof_results = {"theta": np.zeros((nchan, nsubj, 3)),
               "beta": np.zeros((nchan, nsubj, 3)),
               "alpha": np.zeros((nchan, nsubj, 3)),
               "gamma": np.zeros((nchan, nsubj, 3))}

band_keys = list(bands.bands.keys())
all_peak_params = []

all_chidx = np.arange(272)
######## Load rest and task ACWs, store in a numpy array
for i, i_subj in enumerate(subjs_common):
    restname = f"/BICNAS2/ycatal/erf_acw2/results/int_fooof/rest/{i_subj}_fooof.pkl"

    rest_i_fooof = pklload(restname)
    fg = rest_i_fooof["fm"]

    all_peak_params.append([i.peak_params for i in fg])

    # If there is a missing channel, find it and fill with nans
    missingchan_rest = np.setdiff1d(chlist, rest_i_fooof["chanlist"])
    if len(missingchan_rest) != 0:
        missing_idx_rest = np.where(chlist == missingchan_rest)[0]
        good_idx_rest = np.setdiff1d(all_chidx, missing_idx_rest)

        for j in range(len(band_keys)):
            band_powers[band_keys[j]][missing_idx_rest, i] = np.nan
    else:
        good_idx_rest = all_chidx.copy()

    for band in band_keys:
        power = fooof.analysis.get_band_peak_fg(fg, bands[band])
        band_powers[band][good_idx_rest, i] = power[:, 1]
        band_fooof_results[band][good_idx_rest, i, :] = power

savename = f"/BICNAS2/ycatal/erf_acw2/results/int_fooof/rest_all_powers.pkl"
pklsave(savename, band_powers)
savename = f"/BICNAS2/ycatal/erf_acw2/results/int_fooof/rest_all_fooof_results.pkl"
pklsave(savename, band_fooof_results)

##########################################################################
import pingouin as pg
fooof_power = pklload(f"/BICNAS2/ycatal/erf_acw2/results/int_fooof/rest_alpha_power.pkl")
fooof_power = fooof_power["rest_alpha_power"]
acw_ints = pklload(f"/BICNAS2/ycatal/erf_acw2/results/int/rest_int.pkl")
acw_ints = acw_ints["rest_acw_50s"]
acw_ints = np.nanmean(acw_ints, axis=0)

pg.corr(fooof_power.ravel(), acw_ints.ravel(), method="spearman")

corr_results_r = []
corr_results_p = []
all_corr_results = []
for i in range(fooof_power.shape[0]):
    i_corr_result = pg.corr(fooof_power[i, :], acw_ints[i, :], method="spearman")
    corr_results_r.append(i_corr_result["r"].values[0])
    corr_results_p.append(i_corr_result["p-val"].values[0])
    all_corr_results.append(i_corr_result)

corr_results_r
corr_results_p

for i in range(len(all_corr_results)):
    print(all_corr_results[i])

