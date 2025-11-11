from erf_acw2.src import pklload
import numpy as np

savename = f"/BICNAS2/ycatal/erf_acw2/results/int_fooof/rest_all_r2_2fit.pkl"
band_r2 = pklload(savename)
# band_r2 is a dictionary with keys "low" and "high"
# Each key has a numpy array of shape (nchan, nsubj)
# We want to compute the median r2 for each band across all subjects
# and store the results in a new dictionary

r2_results = {}
for low_high in band_r2.keys():
    r2_results[low_high] = np.nanmedian(band_r2[low_high])

pklsave(f"/BICNAS2/ycatal/erf_acw2/results/int_fooof/rest_all_r2_2fit_mean.pkl", r2_results)
