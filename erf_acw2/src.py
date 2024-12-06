import numpy as np
import pickle
from statsmodels.tsa.stattools import acf
import matplotlib.pyplot as plt
import scipy as sp
from scipy.stats._axis_nan_policy import _axis_nan_policy_factory
from os.path import join as pathjoin
import os
import mne
from pathlib import Path
from erf_acw2.meg_chlist import chlist
import scipy as sp
from mne.filter import next_fast_len
import arviz as az
import matplotlib as mpl


def calc_acw(ts, fs, nlags=None):
    """
    Parameters
    ----------
    ts : 1D Numpy vector (Has to be in the shape (n, ). Otherwise won't work)
        Time series.
    fs : double
        Sampling rate (in Hz).

    Returns
    -------
    acw_50 : Where autocorrelation function reaches 0.5
    acfunc : Autocorrelation function (for troubleshooting / plotting etc.)
    lags : x-axis of ACF, for plotting purposes
    """
    if nlags == None:
        nlags = len(ts)

    acfunc = acf(ts, nlags=nlags - 1, qstat=False, alpha=None, missing="conservative")
    lags = np.arange(0.0, nlags / fs, 1.0 / fs)

    acw_50 = np.argmax(acfunc <= 0.5) / fs
    return acw_50, acfunc, lags


def loop_acw_acf(epochs, nlags=None, ntrial=170, nchan=272):
    fs = epochs.info["sfreq"]
    acw50s = np.zeros((ntrial, nchan))
    acfs = np.zeros((ntrial, nchan, nlags))

    c = 0
    for i in range(ntrial):
        if i in epochs.selection:
            i_data = np.squeeze(epochs[c].get_data(copy=True))
            for j in range(nchan):
                acw50s[i, j], acf, _ = calc_acw(i_data[j, :], fs, nlags=nlags)
                acfs[i, j, :] = acf

            c += 1
        else:
            acw50s[i, :] = np.nan
            acfs[i, j, :] = np.nan

    return acw50s, acfs


def subjs():
    return np.genfromtxt("/BICNAS2/ycatal/erf_acw/erf_acw/subjlist.txt", dtype="str")


def pklsave(fname, obj):
    with open(fname, "wb") as f:
        pickle.dump(obj, f)

    print(f"File saved: {fname}")
    return None


def pklload(fname):
    with open(fname, "rb") as f:
        obj = pickle.load(f)
    return obj


def pick_megchans(epochs):
    # Keep in mind that this is in-place
    picks = np.where(np.asarray(epochs.get_channel_types()) == "mag")[0]
    epochs.load_data().pick(picks=picks)
    return epochs


def auto_threshold_F(X, pval=0.05):
    # Get threshold
    dfn = len(X) - 1  # degrees of freedom numerator
    dfd = X[0].shape[0] - len(X)  # degrees of freedom denominator
    thresh = sp.stats.f.ppf(1 - pval, dfn=dfn, dfd=dfd)  # F distribution
    return thresh


def auto_threshold_t(X, pval=0.05):
    df = X[0].shape[0] - 1  # degrees of freedom for the test
    thresh = sp.stats.t.ppf(1 - pval / 2, df)  # two-tailed, t distribution
    return thresh


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx


def auto_threshold_r(X, pval=0.05):
    n = X[0].shape[0]
    dist = sp.stats.beta(n / 2 - 1, n / 2 - 1, loc=-1, scale=2)
    # Since there is no clsoed formula for threshold, implement a numerical method to get it
    r_candidates = np.arange(-1, 1.01, 0.001)
    p_candidates = 2 * dist.cdf(-abs(r_candidates))
    thresh = np.abs(r_candidates[find_nearest(p_candidates, pval)[1]])
    return thresh


def auto_threshold_rho(X, pval=0.05):
    # Numerically find the threshold for significance for spearman correlation
    n = X[0].shape[0]
    rs = np.arange(-1, 1.0, 0.001)
    t = rs * np.sqrt((n - 2) / ((rs + 1.0) * (1.0 - rs)))
    p_candidates = sp.stats.distributions.t.sf(np.abs(t), n - 2) * 2
    thresh = np.abs(rs[find_nearest(p_candidates, pval)[1]])
    return thresh


# Add axis/nan_policy support to stats.pearsonr
pearsonr = _axis_nan_policy_factory(
    lambda *res: tuple(res), n_samples=2, too_small=1, paired=True
)(sp.stats.pearsonr)

# Same for spearman
spearmanr = _axis_nan_policy_factory(
    lambda *res: tuple(res), n_samples=2, too_small=1, paired=True
)(sp.stats.spearmanr)


def import_exampleraw():
    # import one example raw data
    subj = "sub-ON80038"

    datadir = "/BICNAS2/group-northoff/NIMH_healthy_volunteer"
    preprocpath = "/BICNAS2/group-northoff/NIMH_healthy_volunteer/preprocessing"

    subjdir = os.path.join(datadir, subj, "ses-01", "meg")
    raw = mne.io.read_raw_ctf(
        pathjoin(subjdir, subj + "_ses-01_task-movie_run-01_meg.ds"), preload=False
    )

    raw = pick_megchans(raw)
    return raw


def re_epoch(epochs, new_window_size, conservative=False):
    """
    If conservative, then drop all epochs that contain any nan.
    """
    ntrial = len(epochs.drop_log)
    nchan = len(epochs.info.get_channel_types())
    n_timepoints = epochs.get_data().shape[2]
    all_data = np.zeros((ntrial, nchan, n_timepoints))

    all_data[epochs.selection, :, :] = epochs.get_data()
    all_data[np.setdiff1d(np.arange(ntrial), epochs.selection), :, :] = np.nan

    pseudo_raw_data = np.hstack(all_data)
    re_raw = mne.io.RawArray(pseudo_raw_data, epochs.info)

    reepoched = mne.make_fixed_length_epochs(
        re_raw, duration=new_window_size, preload=True
    )
    # Mark the nans as not selected
    nanidx = np.isnan(reepoched.get_data())
    if conservative:
        reepoched.selection = np.where(~np.any(nanidx, axis=(1, 2)))[0]
    else:
        reepoched.selection = np.where(~np.all(nanidx, axis=(1, 2)))[0]

    return reepoched


# Do we really need this?
# def create_blockacws(acw, task, timeaverage=True, mod="10sec"):
#     """
#     Create the dict blockacws
#     acw: a dict with keys "task_acw_*" where * is ["0s", "50s", "drs"]
#     task: a string (e.g. "haririhammer")
#     if timeaverage, average x time
#     """

#     segments = np.load(
#         "/BICNAS2/ycatal/meg_intdiff/results/erp/haririhammer/haririhammer_cont_segments.npy"
#     )
#     ntp, nchan, nsubj = acw["rest_acw_50s"].shape
#     blocks = ["face", "shape"]

#     acwnames = ["acw50"]
#     blockacws = {}
#     names2 = ["50s"]
#     for i in blocks:
#         if timeaverage:
#             blockacws[i] = {
#                 "acw50": np.zeros((nsubj, nchan)),
#             }
#         else:
#             blockacws[i] = {
#                 "acw50": np.zeros((ntp, nchan, nsubj)),
#             }

#     for i, i_block in enumerate(blocks):
#         for j, j_acw in enumerate(names2):
#             for k in range(nsubj):
#                 good_block = np.array([i_block in l for l in segments[k, :]])
#                 if timeaverage:
#                     blockacws[i_block][acwnames[j]][k, :] = np.nanmean(
#                         acw[f"task_acw_{j_acw}"][good_block, :, k], axis=0
#                     )
#                 else:
#                     blockacws[i_block][acwnames[j]][:, :, k] = acw[f"task_acw_{j_acw}"][
#                         :, :, k
#                     ]

#     return blockacws
#


def badchan_padnan(i_chlist, data, nchan=272):
    """
    data is 1D np.array. Assume n channels are missing, so instead of
    data.shape == nchan, it is nchan-n. Create a zero vector of
    shape == nchan, fill the good channels with data, put np.nans to
    bad chans.
    """

    missingchan = np.setdiff1d(chlist, i_chlist)
    where = []
    for j in missingchan:
        where.append(np.where(chlist == np.array([j])))
    where = np.squeeze(np.array(where))  # fucking mental gymnastics

    goodchan = np.setdiff1d(np.arange(nchan), where)
    paddata = np.zeros(nchan)
    paddata[goodchan] = data
    if where.shape == ():
        badchan = where.item()
        paddata[badchan] = np.nan
    elif len(where) != 0:
        paddata[where] = np.nan

    return paddata


def badchan_padnan_2d(i_chlist, data, nchan=272):
    """
    Same as above but for 2d data (channels x time)
    """

    missingchan = np.setdiff1d(chlist, i_chlist)
    where = []
    for j in missingchan:
        where.append(np.where(chlist == np.array([j])))
    where = np.squeeze(np.array(where))  # fucking mental gymnastics

    ntime = data.shape[1]
    goodchan = np.setdiff1d(np.arange(nchan), where)
    paddata = np.zeros((nchan, ntime))
    paddata[goodchan, :] = data

    if where.shape == ():
        badchan = where.item()
        paddata[badchan, :] = np.nan
    elif len(where) != 0:
        paddata[where, :] = np.nan

    return paddata


def badchan_padnan_3d(i_chlist, data, n_epoch, nchan=272):
    """
    Same as above but for 3d data (epochs x channels x time)
    """

    missingchan = np.setdiff1d(chlist, i_chlist)
    where = []
    for j in missingchan:
        where.append(np.where(chlist == np.array([j])))
    where = np.squeeze(np.array(where))  # fucking mental gymnastics

    ntime = data.shape[2]
    goodchan = np.setdiff1d(np.arange(nchan), where)
    paddata = np.zeros((n_epoch, nchan, ntime))
    paddata[:, goodchan, :] = data

    if where.shape == ():
        badchan = where.item()
        paddata[:, badchan, :] = np.nan
    elif len(where) != 0:
        paddata[:, where, :] = np.nan

    return paddata


def get_commonsubj():
    subjlist = subjs()
    task_bad = np.genfromtxt(
        f"/BICNAS2/ycatal/erf_acw2/erf_acw2/badsubjs_haririhammer.txt",
        dtype="str",
    )

    rest_bad = np.genfromtxt(
        "/BICNAS2/ycatal/erf_acw2/erf_acw2/badsubjs_rest.txt", dtype="str"
    )

    subjs_common = np.setdiff1d(np.setdiff1d(subjlist, rest_bad), task_bad)
    return subjs_common


def badepoch_padnan(epochs):
    ntrial = len(epochs.drop_log)
    nchan = len(epochs.info.get_channel_types())
    n_timepoints = epochs.get_data().shape[2]
    all_data = np.zeros((ntrial, nchan, n_timepoints))

    all_data[epochs.selection, :, :] = epochs.get_data()
    all_data[np.setdiff1d(np.arange(ntrial), epochs.selection), :, :] = np.nan

    return all_data


def annot_stat(star, x1, x2, y, h, col="k", ax=None):
    """
    taken from https://stackoverflow.com/questions/65557702/adding-statistical-significance-annotations-to-barplot-subplots
    y: start of the thick line. The line starts from y and goes up to y+h
    """
    ax = plt.gca() if ax is None else ax
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c=col)
    ax.text((x1 + x2) * 0.5, y + h, star, ha="center", va="bottom", color=col)


def p2s(pvals):
    """
    Convert p values to significance stars
    """
    s = []
    for p in pvals:
        if p < 0.001:
            s.append("***")
        elif p < 0.01:
            s.append("**")
        elif p < 0.05:
            s.append("*")
        elif p >= 0.05:
            s.append("n.s.")

    return s


def add_ps(pvals, x_comparisons, y, h, ax=None):
    """
    pvals is a list of p values
    x_comparisons is a list of comparisons on the x axis of plot
    e.g. [[1,2], [2,3]]
    Ordering of x_comparisons has to be same as pvals
    y and h are same as annot_stat
    """
    ax = plt.gca() if ax is None else ax
    ps = p2s(pvals)

    for i, p in enumerate(ps):
        xs = x_comparisons[i]
        annot_stat(p, xs[0], xs[1], y, h, ax=ax)

    return None


rootpath = "/BICNAS2/group-northoff/NIMH_healthy_volunteer"


def get_subjpath(subj, task="haririhammer"):
    datapath = pathjoin(rootpath, "preprocessing", subj, task, "rejection_ica")
    return datapath


def get_subjdatapath(
    subj, task="haririhammer", continuous=False, preproc=True, emo=False
):
    if preproc:
        datapath = pathjoin(rootpath, "preprocessing", subj, task, "rejection_ica")
        if continuous:
            filename = f"{subj}_{task}_preprocessed_cont-epo.fif.gz"
        else:
            if emo:
                filename = f"{subj}_{task}_preprocessed_emo-epo.fif.gz"
            else:
                filename = f"{subj}_{task}_preprocessed-epo.fif.gz"
    else:
        datapath = pathjoin(rootpath, subj, "ses-01", "meg")
        filename = f"{subj}_ses-01_task-{task}_run-01_meg.ds"

    return pathjoin(datapath, filename)


def p2str(p):
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return ""


# Function to calculate the probability of direction
def p_direction(idata, varname):
    merged = az.extract(idata, var_names=varname)
    if len(merged.shape) == 1:
        return np.max(
            [
                np.mean(np.ravel(idata.posterior[varname]) > 0),
                np.mean(np.ravel(idata.posterior[varname]) < 0),
            ]
        )
    else:
        n_dim = merged.shape[0]
        p_direction = np.zeros(n_dim)
        for i in range(n_dim):
            data = merged.values[i]
            p_direction[i] = np.max(
                [
                    np.mean(data > 0),
                    np.mean(data < 0),
                ],
            )
        return p_direction


# Function to calculate ROPE
def rope(idata, varname, rope_range=0.1, hdi_prob=0.94):
    hdi = az.hdi(idata.posterior[varname], hdi_prob=hdi_prob)
    merged = az.extract(idata, var_names=varname)
    if len(merged.shape) == 1:
        data = np.ravel(idata.posterior[varname])
        data_in_hdi = np.logical_and(
            data >= hdi[varname].values[0], data <= hdi[varname].values[1]
        )
        data_in_rope = np.logical_or(
            data[data_in_hdi] >= -rope_range, data[data_in_hdi] <= rope_range
        )
        return np.mean(data_in_rope)
    else:
        n_dim = merged.shape[0]
        rope_result = np.zeros(n_dim)
        for i in range(n_dim):
            data = merged.values[i]
            data_in_hdi = np.logical_and(
                data >= hdi[varname].values[i][0], data <= hdi[varname].values[i][1]
            )
            data_in_rope = np.logical_or(
                data[data_in_hdi] >= -rope_range, data[data_in_hdi] <= rope_range
            )
            rope_result[i] = np.mean(data_in_rope)
        return rope_result


import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid


def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name="shiftedcmap"):
    """
    Source: https://stackoverflow.com/questions/7404116/defining-the-midpoint-of-a-colormap-in-matplotlib
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero.

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower offset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax / (vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highest point in the colormap's range.
          Defaults to 1.0 (no upper offset). Should be between
          `midpoint` and 1.0.
    """
    cdict = {"red": [], "green": [], "blue": [], "alpha": []}

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack(
        [
            np.linspace(0.0, midpoint, 128, endpoint=False),
            np.linspace(midpoint, 1.0, 129, endpoint=True),
        ]
    )

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict["red"].append((si, r, r))
        cdict["green"].append((si, g, g))
        cdict["blue"].append((si, b, b))
        cdict["alpha"].append((si, a, a))

    newcmap = mpl.colors.LinearSegmentedColormap(name, cdict)
    mpl.colormaps.register(cmap=newcmap)

    return newcmap


def plot_channel_effects(
    data, hdi_data, info, chan_order, vlim, title, figpath, mask=None, cmap="PiYG"
):
    f, ax = plt.subplots(1, 3, layout="constrained", figsize=(15, 5))

    titles = ["3% HDI", "Mean", "97% HDI"]
    plot_data = [hdi_data[chan_order, 0], data[chan_order], hdi_data[chan_order, 1]]

    vmin = vlim[0]
    vmax = vlim[1]
    vnew = np.max([abs(vmin), vmax])
    vlim2 = (-vnew, vnew)

    for i in range(3):
        im, _ = mne.viz.plot_topomap(
            data=plot_data[i],
            pos=info,
            axes=ax[i],
            cmap=cmap,
            vlim=vlim2,
            show=False,
            size=15,
            res=800,
            mask=(
                mask[chan_order] if mask is not None else None
            ),  # Apply the mask if provided
            mask_params=dict(
                marker="x",
                markerfacecolor="blue",
                markeredgecolor="blue",
                linewidth=0,
                markersize=7,
            ),
        )
        ax[i].set_title(titles[i])

    cb = f.colorbar(im, ax=ax.ravel().tolist(), fraction=0.05, pad=0.04)
    cb.ax.set_ylim([vlim[0], vlim[1]])
    f.suptitle(title)

    plt.savefig(figpath, dpi=500)
    plt.close(f)


# Create mask from ROPE values
def create_channel_mask(summary, var_name):
    # Get all rows that correspond to the channel effects
    # Use exact match with [] to avoid matching similar variable names
    channel_rows = summary[summary.index.str.startswith(f"{var_name}[")]
    # Create mask where ROPE <= 0.01
    mask = channel_rows["rope"].values <= 0.01
    return mask
