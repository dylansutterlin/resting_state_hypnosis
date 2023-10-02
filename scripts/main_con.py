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
from nilearn import datasets, plotting, image
from nilearn.image import concat_imgs
from nilearn.regions import connected_label_regions
from nilearn.signal import clean as signal_clean
from src import glm_func, graphs_regressionCV
from scripts import func
from src import masker_preprocessing as prep


def con_matrix(
    data_dir,
    conf_dir,
    cwd,
    save_folder=None,
    save_base=None,
    atlas_name=None,
    sphere_coord=None,
    connectivity_measure="correlation",
    plot_atlas=False,
    verbose=False,
):
    """_summary_

    Parameters
    ----------
    data_dir : str
        Path to data, fMRI images (.hdr or .nii)
    conf_dir : str
        Path to each subject's folder containing regressors and confounds
        **This arg was added to account for data structure, e.i. fMRI imgages stored in diff folder than regressors and other data and
        for some permission reasons on the server, it wasn't possible to move the confound file, could be fixed in the futur!**
    save_base : str
        Path to saving folder
    save_folder : str
        Name of folder/atlas/condition to name the folder in which results will be saved, e.g. 'yeo_7'
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
    data = func.load_data(data_dir, conf_dir)
    conditions = ["pre_hyp", "post_hyp", "contrast"]
    pre_data, post_data = prep.resample_shape_affine(data)
    results = dict(pre_series=list(), post_series=list())
    results["subjects"] = data.subjects
    all_data = pre_data + post_data
    # --Atlas choices--
    atlas, atlas_labels, atlas_type, confounds = func.load_choose_atlas(
        atlas_name, cwd, bilat=True
    )
    print("atlas done!")
    # basic masker
    voxel_masker = MultiNiftiMasker(
        mask_strategy="whole-brain-template",
        high_pass=0.1,
        t_r=3,
        standardize="zscore_sample",
        verbose=5,
    )
    # (prep.check_masker_fit(da, voxel_masker) for da in [pre_data, post_data])

    # transf_imgs, fitted_voxel_masker, brain_mask = glm_func.transform_imgs(
    #    all_files, voxel_masker, return_series=False
    # )

    # --ROI masker parameters--
    if atlas_name == None:
        masker = prep.choose_tune_masker(use_atlas_type=False, mask_img=False)
    else:
        masker = prep.choose_tune_masker(use_atlas_type=atlas_name, mask_img=False)

    # --Timeseries : Fit and apply mask--
    print(masker)
    masker.fit(all_data)
    results["pre_series"] = [
        masker.transform(ts, confounds=conf)
        for ts, conf in zip(pre_data, data.confounds_pre_hyp)
    ]
    results["post_series"] = [
        masker.transform(ts, conf)
        for ts, conf in zip(post_data, data.confounds_post_hyp)
    ]

    # --Seed masker--
    if sphere_coord != None:
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
        # masker = NiftiMasker(mask_img=atlas, standardize=True)
        # masker.fit(concat_imgs(pre_data))

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
        display.add_markers(
            marker_coords=sphere_coord, marker_color="g", marker_size=300
        )
        # display.savefig("OP_seed_correlation.pdf")

    # Regression
    xlsx_path = (
        r"/data/rainville/HYPNOSIS_ASL_RAW_DATA/Hypnosis_variables_20190114_pr_jc.xlsx"
    )
    xlsx_path = r"C:\Users\Dylan\Desktop\UM_Bsc_neurocog\E22\Projet_Ivado_rainvillelab\test_dataset\test_data_ASL\Hypnosis_variables_20190114_pr_jc.xlsx"
    Y, target_columns = graphs_regressionCV.load_process_y(xlsx_path, data.subjects)
    X_ls, metrics_names = graphs_regressionCV.graph_metrics(results, Y, labels)
    result_regression = graphs_regressionCV.regression_cv(X_ls, metrics_names)

    # --Save--
    if save_base != None:
        if os.path.exists(os.path.join(save_base, save_folder)) is False:
            os.mkdir(os.path.join(save_base, save_folder))
        save_to = os.path.join(save_base, save_folder)

        func.save_results(data.subjects, save_to, conditions, results)
        results = func.extract_features(
            results
        )  # apply trilower mask and vectorize connectomes
        func.npsave_features(save_to, results)

        with open(os.path.join(save_to, "dict_results.pkl"), "wb") as f:
            pickle.dump(results, f)
        with open(os.path.join(save_to, "dict_regression.pkl"), "wb") as f:
            pickle.dump(result_regression, f)
        print("Saved result dict!")

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
