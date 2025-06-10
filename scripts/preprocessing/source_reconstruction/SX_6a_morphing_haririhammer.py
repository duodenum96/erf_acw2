import mne
import numpy as np
import os
from os.path import join as pathjoin
import sys
from erf_acw2.src import get_commonsubj, pklload, pklsave
os.chdir("/BICNAS2/ycatal/erf_acw2/scripts/preprocessing")
from scripts.preprocessing.rest_badICs import exclude, badics

subjects_dir = "/BICNAS2/group-northoff/NIMH_source_reconstruction"
fsaverage_bem_path = "/BICNAS2/group-northoff/NIMH_source_reconstruction/fsaverage_bem"

fsaverage_src_path = pathjoin(fsaverage_bem_path, "fsaverage-ico-4-src.fif")
if not os.path.exists(fsaverage_src_path):
    src = mne.setup_source_space('fsaverage', spacing='ico4', subjects_dir=subjects_dir)
    mne.write_source_spaces(fsaverage_src_path, src)


fname_fsaverage_src = pathjoin(fsaverage_bem_path, "fsaverage-ico-4-src.fif")

surfer_kwargs = dict(
    hemi="lh",
    subjects_dir=subjects_dir,
    clim=dict(kind="value", lims=[8, 12, 15]),
    views="lateral",
    initial_time=0.09,
    time_unit="s",
    size=(800, 800),
    smoothing_steps=5,
)

i = int(sys.argv[1])
subjlist = get_commonsubj()
i_subj = subjlist[i]

if i_subj in exclude:
    sys.exit("Bad Subject")

# Setup data
data_root = "/BICNAS2/group-northoff/NIMH_healthy_volunteer"
subj_raw_path = pathjoin(data_root, i_subj, "ses-01", "meg")

preprocpath = "/BICNAS2/group-northoff/NIMH_healthy_volunteer/preprocessing"

subj_preprocpath = pathjoin(preprocpath, i_subj)
outputpath = pathjoin(subj_preprocpath, "haririhammer", "morphing")

if not os.path.exists(outputpath):
    os.makedirs(outputpath)

fwd = mne.read_forward_solution(pathjoin(subj_preprocpath, "haririhammer", "forward-fwd.fif"))
src = fwd["src"]

src_to = mne.read_source_spaces(fname_fsaverage_src)

tasknames = ["encode_face_happy", "probe_face_happy", "encode_face_sad", "probe_face_sad", "probe_shape", "encode_shape"]

for i_taskname in tasknames:
    filename = pathjoin(subj_preprocpath, "haririhammer", f"{i_taskname}_stc.pkl")
    stc = pklload(filename)
    morph = mne.compute_source_morph(
        stc,
        subject_from=i_subj,
        subject_to="fsaverage",
        src_to=src_to,
        subjects_dir=subjects_dir,
    )
    stc_fsaverage = morph.apply(stc)
    # brain_inf = stc_fsaverage.plot(surface="inflated", **surfer_kwargs)
    morph.save(pathjoin(outputpath, f"{i_taskname}_morph"), overwrite=True)

print(f"Subject {i_subj} done")
