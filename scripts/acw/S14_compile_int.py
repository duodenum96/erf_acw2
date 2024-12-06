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

nlags = 2000
rest_acw_50s = np.zeros((rest_ntp, nchan, nsubj))
acfs = np.zeros((rest_ntp, nchan, nsubj, nlags))

all_chidx = np.arange(272)
######## Load rest and task ACWs, store in a numpy array
for i, i_subj in enumerate(subjs_common):
    restname = f"/BICNAS2/ycatal/erf_acw2/results/int/rest/{i_subj}_rest_acw.pkl"

    rest_i_acw = pklload(restname)

    # If there is a missing channel, find it and fill with nans
    missingchan_rest = np.setdiff1d(chlist, rest_i_acw["chanlist"])
    if len(missingchan_rest) != 0:
        missing_idx_rest = np.where(chlist == missingchan_rest)[0]
        good_idx_rest = np.setdiff1d(all_chidx, missing_idx_rest)

        rest_acw_50s[:, missing_idx_rest, i] = np.nan
        acfs[:, missing_idx_rest, i, :] = np.nan
    else:
        good_idx_rest = all_chidx.copy()

    rest_acw_50s[:, good_idx_rest, i] = rest_i_acw["acw50"]
    acfs[:, good_idx_rest, i, :] = rest_i_acw["acf"]

allacws = {"rest_acw_50s": rest_acw_50s, "acfs": acfs}

savename = f"/BICNAS2/ycatal/erf_acw2/results/int/rest_int.pkl"
pklsave(savename, allacws)
