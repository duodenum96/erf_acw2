import mne
import os 
from os.path import join as pathjoin
from erf_acw2.src import get_commonsubj
import sys
from mne.coreg import Coregistration
os.chdir("/BICNAS2/ycatal/erf_acw2/scripts/preprocessing")
from scripts.preprocessing.rest_badICs import exclude, badics
import numpy as np

subjects_dir = "/BICNAS2/group-northoff/NIMH_source_reconstruction"

i = int(sys.argv[1])
subjlist = get_commonsubj()
i_subj = subjlist[i]

plot_kwargs = dict(
    subject=i_subj,
    subjects_dir=subjects_dir,
    surfaces="head-dense",
    dig=True,
    eeg=[],
    meg="sensors",
    show_axes=True,
    coord_frame="meg",
)

if i_subj in exclude:
    sys.exit("Bad Subject")

data_root = "/BICNAS2/group-northoff/NIMH_healthy_volunteer"
subj_raw_path = pathjoin(data_root, i_subj, "ses-01", "meg")

preprocpath = "/BICNAS2/group-northoff/NIMH_healthy_volunteer/preprocessing"

subj_preprocpath = pathjoin(preprocpath, i_subj)
outputpath = pathjoin(subj_preprocpath, "rest", "rejection_ica")

data = mne.read_epochs(pathjoin(outputpath, i_subj + "_rest_preprocessed-epo.fif.gz"))

bem_path = f"/BICNAS2/group-northoff/NIMH_source_reconstruction/{i_subj}/bem"
if not os.path.isdir(bem_path):
    mne.bem.make_watershed_bem(  # for T1; for FLASH, use make_flash_bem instead
        subject=i_subj,
        subjects_dir=subjects_dir,
        copy=True,
        overwrite=True,
    )

coreg = Coregistration(data.info, i_subj, subjects_dir)

# fig = mne.viz.plot_alignment(data.info, trans=coreg.trans, **plot_kwargs)
# fig.savefig(pathjoin(subj_preprocpath, "rest", "alignment_inititial.png"))

coreg.fit_fiducials(verbose=True)
# fig = mne.viz.plot_alignment(data.info, trans=coreg.trans, **plot_kwargs)
# fig.savefig(pathjoin(subj_preprocpath, "rest", "alignment_fiducial_fitted.png"))

coreg.fit_icp(n_iterations=6, nasion_weight=2.0, verbose=True)
# fig = mne.viz.plot_alignment(data.info, trans=coreg.trans, **plot_kwargs)
# fig.savefig(pathjoin(subj_preprocpath, "rest", "alignment_icp_fitted.png"))

coreg.omit_head_shape_points(distance=5.0 / 1000)  # distance is in meters
# fig = mne.viz.plot_alignment(data.info, trans=coreg.trans, **plot_kwargs)
# fig.savefig(pathjoin(subj_preprocpath, "rest", "alignment_omit_head_shape_points.png"))

coreg.fit_icp(n_iterations=20, nasion_weight=10.0, verbose=True)
# fig = mne.viz.plot_alignment(data.info, trans=coreg.trans, **plot_kwargs)
# fig.savefig(pathjoin(subj_preprocpath, "rest", "alignment_icp_fitted_20.png"))

dists = coreg.compute_dig_mri_distances() * 1e3  # in mm
print(
    f"Distance between HSP and MRI (mean/min/max):\n{np.mean(dists):.2f} mm "
    f"/ {np.min(dists):.2f} mm / {np.max(dists):.2f} mm"
)

mne.write_trans(pathjoin(subj_preprocpath, "rest", "alignment-trans.fif"), coreg.trans)
