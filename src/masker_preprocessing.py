import os
import numpy as np
import pandas as pd
from nilearn.maskers import (
    MultiNiftiLabelsMasker,
    MultiNiftiMapsMasker,
    NiftiLabelsMasker,
    NiftiMapsMasker,
    NiftiMasker,
    NiftiSpheresMasker,
)
from nilearn.image import concat_imgs, mean_img, resample_to_img
from nilearn.plotting import plot_epi, plot_roi, plot_stat_map
from scripts import func


def choose_atlas_masker(
    atlas,
    atlas_type,
    mask_img=None,
    tr=None,
    smoothing_fwhm=None,
    standardize=False,
    verbose=5,
    resampling_target='data',
    confounds=None,
):
    """
    Choose and tune masker parameters
    Parameters
    ----------
    atlas : NiftiLabelsMasker or NiftiMapsMasker object
    atlas_type : str or bool
        Choices : 'labels' and 'maps' for probabilistic atlases, by default False
    mask_img : str, optional
        Path to mask image, by default None
    tr : int, optional
        Repetition time, by default None
    smoothing_fwhm : int, optional
        Smoothing kernel, by default None
    standardize : str, optional
        Standardization method, by default None
    verbose : int, optional
        Verbosity level, by default None
    resampling_target : str, optional
        Resampling target, by default None
    atlas_type : str, optional
        Atlas type, by default None
    confounds : str, optional
        Confounds, by default None

    Returns
    -------
    masker : nilearn.maskers.NiftiMasker object. Not fitted
    """

    if atlas_type == "maps":
        masker = NiftiMapsMasker(
            maps_img=atlas,
            mask_img= mask_img,
            t_r=tr,
            smoothing_fwhm=smoothing_fwhm,
            standardize=standardize,
            verbose=verbose,
            resampling_target=resampling_target,
        )
        print("Probabilistic (maps) atlas selected!")
    elif atlas_type == "labels":
        # labels = atlas.labels
        masker = MultiNiftiLabelsMasker(
            labels_img=atlas,
            labels=atlas_labels,
            standardize=standardize,
            resampling_target=resampling_target,
        )
        print("Label (not probabilistic) masker!")
    else:
        raise ValueError("Atlas type must be maps or labels in order to choose NiftiMasker!")
    
        
    return masker


def check_masker_fit(data, masker):
    # print basic information on the dataset
    print("First functional nifti image (4D) is located " f"at: {data}")

    filename = data
    mean_im = mean_img(filename)
    plot_epi(mean_im, title="Mean EPI image")

    masker.fit(data)
    print("Masker fit done, see html report!")
    report = masker.generate_report()
    report.save_as_html("masker_report.html")

    # plot the mask
    plot_roi(masker.mask_img_, mean_im, title="Mask")


def resample_shape_affine(data, target_img="first_data"):
    resampled = []
    split = len(data.func_pre_hyp)
    all_files = data.func_pre_hyp + data.func_post_hyp
    if target_img == "first_data":
        ref_img = mean_img(all_files)

    for i in range(len(all_files)):
        resampled.append(
            resample_to_img(all_files[i], ref_img, interpolation="continuous")
        )

    return resampled[:split], resampled[split:]


def transform_imgs(all_files, masker, return_series=False):
    masker.fit(all_files)
    voxel_series = [masker.transform(ts) for ts in all_files]
    trans_imgs = [masker.inverse_transform(ts) for ts in voxel_series]
    if return_series:
        return trans_imgs, masker, voxel_series
    else:
        return trans_imgs, masker, masker.mask_img_
