#!/bin/bash

# Run freesurfer recon-all
cd /BICNAS2/ycatal/erf_acw2/scripts/preprocessing

subjlist_file="/BICNAS2/ycatal/erf_acw2/erf_acw2/commonsubj.txt"
subjlist=($(cat $subjlist_file))

mri_data_root=/BICNAS2/group-northoff/NIMH_healthy_volunteer

n_threads=16

i=$1

subj=${subjlist[$i]}
echo "Running freesurfer for $subj"
subj_root=${mri_data_root}/${subj}/ses-01/anat
file_fsgpr=${subj}_ses-01_acq-FSPGR_rec-SCIC_T1w.nii.gz
file_mprage=${subj}_ses-01_acq-MPRAGE_rec-SCIC_T1w.nii.gz

if [ -f "${subj_root}/${file_fsgpr}" ]; then
    echo "FSPGR file found for $subj"
    file=$file_fsgpr
elif [ -f "${subj_root}/${file_mprage}" ]; then
    echo "MPRAGE file found for $subj"
    file=$file_mprage
else
    echo "Neither FSPGR nor MPRAGE file found for $subj"
    continue
fi

path=${subj_root}/${file}
recon-all -s $subj -i $path -all -parallel -openmp $n_threads

echo "Freesurfer completed for $subj"

