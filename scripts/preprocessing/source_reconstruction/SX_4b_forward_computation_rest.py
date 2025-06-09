import mne
import numpy as np
import os
from os.path import join as pathjoin
import sys
from erf_acw2.src import get_commonsubj
os.chdir("/BICNAS2/ycatal/erf_acw2/scripts/preprocessing")
from scripts.preprocessing.rest_badICs import exclude, badics

subjects_dir = "/BICNAS2/group-northoff/NIMH_source_reconstruction"

i = int(sys.argv[1])
subjlist = get_commonsubj()
i_subj = subjlist[i]

if i_subj in exclude:
    sys.exit("Bad Subject")

plot_bem_kwargs = dict(
    subject=i_subj,
    subjects_dir=subjects_dir,
    brain_surfaces="white",
    orientation="coronal",
    slices=[50, 100, 150, 200],
)

# Setup data
data_root = "/BICNAS2/group-northoff/NIMH_healthy_volunteer"
subj_raw_path = pathjoin(data_root, i_subj, "ses-01", "meg")

preprocpath = "/BICNAS2/group-northoff/NIMH_healthy_volunteer/preprocessing"

subj_preprocpath = pathjoin(preprocpath, i_subj)
outputpath = pathjoin(subj_preprocpath, "rest", "rejection_ica")

data = mne.read_epochs(pathjoin(outputpath, i_subj + "_rest_preprocessed-epo.fif.gz"))

# Visualize BEM
fig = mne.viz.plot_bem(**plot_bem_kwargs)
fig.savefig(pathjoin(subj_preprocpath, "rest", "bem.png"))

# Set the transformation matrix path
trans = pathjoin(subj_preprocpath, "rest", "alignment-trans.fif")

# Setup source space
src = mne.setup_source_space(
    i_subj, spacing="oct6", subjects_dir=subjects_dir
)
print(src)

# Visualize BEM with source space
fig = mne.viz.plot_bem(src=src, **plot_bem_kwargs)
fig.savefig(pathjoin(subj_preprocpath, "rest", "bem_oct6_src.png"))

# fig = mne.viz.plot_alignment(
#     subject=i_subj,
#     subjects_dir=subjects_dir,
#     surfaces="white",
#     coord_frame="mri",
#     src=src,
# )
# mne.viz.set_3d_view(
#     fig,
#     azimuth=173.78,
#     elevation=101.75,
#     distance=0.30,
#     focalpoint=(-0.03, -0.01, 0.03),
# )
# fig.savefig(pathjoin(subj_preprocpath, "rest", "bem_oct6_src_alignment.png"))

# Compute forward solution
# Forward solution requires a BEM model
model = mne.make_bem_model(
    subject=i_subj, subjects_dir=subjects_dir
)
bem = mne.make_bem_solution(model)

# Now the forward solution can be computed aka gain aka leadfield matrix
fwd = mne.make_forward_solution(
    data.info,
    trans=trans,
    src=src,
    bem=bem,
    meg=True,
    eeg=False,
    mindist=5.0,
    n_jobs=None,
    verbose=True,
)

print(f"Before: {src}")
print(f'After:  {fwd["src"]}')

leadfield = fwd["sol"]["data"]
print(f"Leadfield size : {leadfield.shape[0]} sensors x {leadfield.shape[1]} dipoles")

mne.write_forward_solution(pathjoin(subj_preprocpath, "rest", "forward-fwd.fif"), fwd)