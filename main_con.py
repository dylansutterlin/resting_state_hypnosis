import argparse
import pickle
import os
import glob
import numpy as np
import nibabel as nib
from nilearn.maskers import (
    NiftiMapsMasker,
    NiftiLabelsMasker,
    NiftiMasker,
    NiftiSpheresMasker,
    MultiNiftiMasker,
    MultiNiftiMapsMasker,
)
from nilearn.connectome import ConnectivityMeasure
import func
from nilearn import datasets, plotting, image
from nilearn.image import concat_imgs
from nilearn.regions import connected_label_regions
from nilearn.signal import clean as signal_clean


def con_matrix(
    data_dir,
    save_folder=None,
    save_base=None,
    atlas_name="yeo_7",
    atlas_type="labels",
    sphere_coord=None,
    connectivity_measure="correlation",
    plot_atlas=False,
    verbose=False,
):
    """_summary_

    Parameters
    ----------
    data_dir : str
        Path to data
    save_base : str
        Path to saving folder
    path_to_atlas : _type_, optional
        Path to atlas, by default None
    atlas_type : str, optional
        Choices : 'labels' and 'maps' for probabilistic atlases, by default 'labels'
    connectivity_measure : str, optional
        Correlation estimation, by default 'correlation'
    plot_atlas : bool, optional
        by default False
    verbose : bool, optional
        Wether to print/plot outputs, by default False
    """

    # --Data--
    data = func.load_data(data_dir)
    conditions = ["pre_hyp", "post_hyp", "contrast"]
    pre_data = data.pre_hyp
    post_data = data.post_hyp
    all_data = pre_data + post_data
    results = dict(pre_series=list(), post_series=list())

    # --Atlas choices--
    atlas, atlas_labels, confounds = func.load_choose_atlas(atlas_name, bilat=True)

   
    # --Labels--
    # region_labels = connected_label_regions(atlas)
    '''
    img_ref = atlas
    target_affine = img_ref.affine
    reference_image = atlas
    all_files = pre_data + post_data
    resampled_images = []
    for image_path in all_files:
        im = nib.load(image_path)
        resampled_image = image.resample_img(
            im, target_affine=target_affine, target_shape=reference_image.shape[:-1]
        )
        resampled_images.append(resampled_image)
    '''

    breakpoint()
    voxel_masker = MultiNiftiMasker(
        mask_strategy="whole-brain-template",
        high_pass=0.1,
        t_r=3,
        standardize=True,
        smoothing_fwhm=6,
        verbose=5,
    )

    check_masker_fit(data, voxel_masker)

    breakpoint()
    # --Masker parameters--
    if atlas_type == "maps":
        masker = MultiNiftiMapsMasker(
            maps_img=atlas,
            mask_img=voxel_masker.mask_img_,
            t_r=3,
            smoothing_fwhm=6,
            standardize="zscore_sample",
            verbose=5,
            resampling_target="maps",
        )
        print("Probabilistic atlas!")
    elif atlas_type == "labels":
        # labels = atlas.labels
        masker = NiftiLabelsMasker(
            labels_img=atlas,
            labels=atlas_labels,
            standardize="psc",
            resampling_target="data",
        )
        print(" Labeled masker!")

    elif atlas_type == None:
        voxel_masker = MultiNiftiMasker(
            mask_strategy="whole-brain-template",
            high_pass=0.1,
            t_r=3,
            standardize=True,
            smoothing_fwhm=6,
            verbose=5,
        )
        # mask_strategy="whole-brain-template"
    # --Timeseries : Fit and apply mask--
    if atlas_type == None:
        all_files = pre_data + post_data
        masker.fit(all_files)

        results["pre_series"] = [masker.transform(ts) for ts in pre_data]
        # masker.fit(image.mean_img(post_data))
        results["post_series"] = [masker.transform(ts) for ts in post_data]

        breakpoint()

    if atlas_type != None:
        all_files = pre_data + post_data
        masker.fit(all_files)
        results["pre_series"] = [masker.transform(ts) for ts in pre_data]
        # masker.fit(post_data)
        results["post_series"] = [masker.transform(ts) for ts in post_data]

    if sphere_coord != None:
        sphere_coord = [(54, -28, 26)]
        seed_masker = NiftiSpheresMasker(
            sphere_coord, radius=8, standardize="zscore_sample"
        )

        results["seed_pre_series"] = [seed_masker.fit_transform(ts) for ts in pre_data]
        results["seed_post_series"] = [
            seed_masker.fit_transform(ts) for ts in post_data
        ]

    # Compute seed-to-voxel correlation
    results["seed_to_pre_correlations"] = [
        (np.dot(brain_time_series.T, seed_time_series) / seed_time_series.shape[0])
        for brain_time_series, seed_time_series in zip(
            results["pre_series"], results["seed_pre_series"]
        )
    ]
    results["seed_to_post_correlations"] = [
        (np.dot(brain_time_series.T, seed_time_series) / seed_time_series.shape[0])
        for brain_time_series, seed_time_series in zip(
            results["post_series"], results["seed_post_series"]
        )
    ]

    # -- Covariance Estimation--
    correlation_measure = ConnectivityMeasure(
        kind=connectivity_measure, discard_diagonal=True
    )
    results = func.compute_cov_measures(correlation_measure, results)

    if sphere_coord != None:
        results["mean_seed_pre_connectome"] = np.mean(
            results["seed_to_pre_correlations"], axis=0
        )
        results["mean_seed_post_connectome"] = np.mean(
            results["seed_to_post_correlations"], axis=0
        )
        results["mean_seed_contrast_connectome"] = (
            results["mean_seed_post_connectome"] - results["mean_seed_pre_connectome"]
        )

    # --Plot--
    from nilearn import plotting

    # masker = NiftiMasker(mask_img=atlas, standardize=True)
    # masker.fit(concat_imgs(pre_data))
    breakpoint()
    seed_to_voxel_correlations_img = masker.inverse_transform(
        results["mean_seed_contrast_connectome"].T
    )

    display = plotting.plot_stat_map(
        seed_to_voxel_correlations_img,
        threshold=0.5,
        vmax=1,
        cut_coords=sphere_coord[0],
        title="Seed-to-voxel correlation (OP seed)",
    )
    display.add_markers(marker_coords=sphere_coord, marker_color="g", marker_size=300)
    # At last, we save the plot as pdf.
    display.savefig("OP_seed_correlation.pdf")

    # --Save--
    if save_base != None:
        if os.path.exists(os.path.join(save_base, save_folder)) is False:
            os.mkdir(os.path.join(save_base, save_folder))
        save_to = os.path.join(save_base, save_folder)
        func.save_results(data.subjects, save_to, conditions, results)

        # --Stats--
        matrices = np.asarray(
            [
                np.load(
                    os.path.join(save_to, f"{sub}_{conditions[2]}_connectomes.npy"),
                    allow_pickle=True,
                )
                for sub in data.subjects
            ]
        )
        y_full_auto = data.phenotype[
            "Unnamed: 68"
        ]  # abs. diff. in perceived automaticity
        # Access selected sub based on id in y
        rename_sub = [f"APM{num}" for num in [sub[4:6] for sub in data.subjects]]
        idcs = [
            idcs
            for idcs, index in enumerate(data.phenotype.index)
            if index in rename_sub
        ]
        y_auto = np.array(y_full_auto[idcs])

        # --X/features (vectorize each connectome)--
        results = func.extract_features(results)

        np.save(
            os.path.join(save_to, f"features_pre"), results["preX"], allow_pickle=True
        )
        np.save(
            os.path.join(save_to, f"features_post"), results["postX"], allow_pickle=True
        )
        np.save(
            os.path.join(save_to, f"features_contrast"),
            results["contrastX"],
            allow_pickle=True,
        )
        np.save(os.path.join(save_to, f"Y"), y_auto, allow_pickle=True)

        with open(os.path.join(save_to, "dict_results.pkl"), "wb") as f:
            pickle.dump(results, f)
        print("Saved pickle dump!")

    # --Prints and plot--
    if verbose:
        print([ts.shape for ts in results["pre_series"]])
        print([ts.shape for ts in results["post_series"]])
        print(atlas.shape)
        print(np.unique(atlas.get_fdata(), return_counts=True))
        for correlation_matrix in [
            results["pre_mean_connectome"],
            results["post_mean_connectome"],
            results["zcontrast_mean_connectome"],
        ]:  # [results['pre_mean_connetomes'], results['post_mean_connetomes']]:
            np.fill_diagonal(correlation_matrix, 0)
            plotting.plot_matrix(
                correlation_matrix,
                labels=atlas_labels,
                colorbar=True,
                vmax=0.8,
                vmin=-0.8,
            )
            func.plot_bilat_nodes(correlation_matrix, atlas, atlas_name)

        plotting.plot_roi(atlas, title=atlas_name)

    return results


"""
# plot connectivity matrix

    matrix = np.load(os.path.join('data/derivatives/connectomes', os.listdir('data/derivatives/connectomes')[0]))
    plotting.plot_matrix(squareform(matrix), vmin = -1, vmax = 1, labels=masker.labels_)
    plt.savefig('results/plots/connectivity_matrix.svg', format='svg')
    plt.clf()
"""


"""
gb_signal = signal_clean(
        np.array(results["pre_series"])
        .mean(axis=1)
        .reshape([np.array(results["pre_series"]).shape[0], 1]),
        high_pass=0.1,
        t_r=3,
        standardize="zscore_sample",
    )

    results["pre_series"] = voxel_masker.fit_transform(pre_data, confounds=gb_signal)
    gb_signal = signal_clean(
        results["post_series"]
        .mean(axis=1)
        .reshape([results["post_series"].shape[0], 1]),
        high_pass=0.1,
        t_r=3,
        standardize="zscore_sample",
    )
    results["post_series"] = voxel_masker.fit_transform(post_data, confounds=gb_signal)
"""
