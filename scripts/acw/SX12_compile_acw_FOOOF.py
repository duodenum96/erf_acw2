import os
import numpy as np
from fooof import FOOOF
import time
import sys

os.chdir("/BICNAS2/ycatal/erf_acw2/scripts/preprocessing")
from scripts.preprocessing.rest_badICs import exclude, badics
from scipy import signal
import matplotlib.pyplot as plt
from erf_acw2.src import pklload, pklsave, pick_megchans, re_epoch, get_commonsubj, pathjoin
import pandas as pd

subjects_dir = "/BICNAS2/group-northoff/NIMH_source_reconstruction"
preprocpath = "/BICNAS2/group-northoff/NIMH_healthy_volunteer/preprocessing"

results_path = "/BICNAS2/ycatal/erf_acw2/results/source"
if not os.path.exists(results_path):
    os.makedirs(results_path)

all_ints = []
subjs = []
vertices = []

subjlist = get_commonsubj()
for i_subj in subjlist:
    if i_subj in exclude:
        continue

    subj_preprocpath = pathjoin(preprocpath, i_subj)
    source_fooof = pklload(pathjoin(subj_preprocpath, "rest", f"{i_subj}_source_fooof.pkl"))
    ints = 1.0 / (2.0 * np.pi * source_fooof["ints"])
    for i_vertex in range(len(ints)):
        all_ints.append(ints[i_vertex])
        vertices.append(i_vertex)
        subjs.append(i_subj)


df = pd.DataFrame({"subjs": subjs, "vertices": vertices, "ints": all_ints})

df.to_csv(pathjoin(results_path, "acw_FOOOF_source.csv"), index=False)



