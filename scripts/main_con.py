import sys
import argparse
import pickle
import os
import copy
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
    pwd_main,
    conf_dir=False,
    save_base=None,
    save_folder=None,
    atlas_name=None,
    sphere_coord=None,
    connectivity_measure="correlation",
    plot_atlas=False,
    n_sub = None,
    verbose=False,
):
    """_summary_

    Parameters
    ----------

    data_dir : str
        Path to data, fMRI images (.hdr or .nii)

    pwd_main : str
        Directory where main.py is ran from. Used to access files likes phenotype and atlases files.

    conf_dir : str
        Path to each subject's folder containing regressors and confounds
        **This arg was added to account for data structure, e.i. fMRI imgages stored in diff folder than regressors and other data and
        for some permission reasons on the server, it wasn't possible to move the confound file, could be fixed in the futur!**

    save_base : str
        Path to saving folder. If None, will be automatically generated in pwd_main

    save_folder : str
        Name of folder/atlas/condition to name the folder in which results will be saved, e.g. 'yeo_7'

    atlas_name : str, optional
        Atlas name to use. If None, yeo7 will be used. Other choices include

    atlas_type : str, optional
        Choices : 'labels' and 'maps' for probabilistic atlases, by default 'labels'

    connectivity_measure : str, optional
        Correlation estimation, by default 'correlation'

    plot_atlas : bool, optional
        by default False

    verbose : bool, optional
        Wether to print/plot outputs, by default False

    Returns
    -------
    results_con : dict
        Dictionary containing all connectivity results, including connectomes, connectome metrics, etc.
    """

    print("---LOADING DATA---")
    data = func.load_data(data_dir, conf_dir, pwd_main, n_sub=n_sub)
    conditions = ["pre_hyp", "post_hyp", "contrast"]
    #pre_data, post_data = prep.resample_shape_affine(data)
    pre_data = data.func_pre
    post_data = data.func_post
    results_con = dict(subjects=data.subjects, pre_series=list(), post_series=list())
    results_graph = dict(pre=dict(), post=dict(), change=dict())
    results_pred = dict(pre=dict(), post=dict(), change=dict())
    
    print('----ATLAS SELECTION---')
    if atlas_name != None:
        atlas, atlas_labels, atlas_type, confounds = func.load_choose_atlas(
            pwd_main, atlas_name, bilat=True
        ) # confounds is not used further

    print('---MASKER SELECTION---')
    voxel_masker = MultiNiftiMasker(
        mask_strategy="whole-brain-template",
        standardize="zscore_sample",
        verbose=5,
    )
    # Return NiftiMaker object yet to be fitted
    if atlas_name == None:
        masker = voxel_masker
    else:
        masker = prep.choose_atlas_masker(
            atlas = atlas, atlas_type =atlas_type, mask_img=None, resampling_target= 'data'
        ) 

    print('---TIMESERIES EXTRACTION---') 
    # --Timeseries : Fit and apply mask--
    p = os.path.join(pwd_main, 'debug','fitted_timeSeries.pkl')
    input_timeseries=False
    if input_timeseries != False:
        with open(os.path.join(p,'fitted_timeSeries.pkl'), 'rb') as f:
            load_results = pickle.load(f)
        results_con['pre_series'] = load_results['pre_series']
        results_con['post_series'] = load_results['post_series']
    else:
        masker.fit(pre_data)
        results_con["pre_series"] = [masker.transform(ts) for ts in pre_data]
        masker.fit(post_data)
        results_con["post_series"] = [masker.transform(ts) for ts in post_data]

        print('---MAKING MASKER REPORTS---')
        voxel_masker.fit(pre_data + post_data)
        voxel_masker_report = voxel_masker.generate_report()
        voxel_masker_report.save_as_html(os.path.join('debug/reports', 'voxel_masker_report.html'))

        nib.save(atlas, os.path.join(pwd_main, 'debug', 'atlas.nii.gz'))
        nib.save(image.mean_img(pre_data + post_data), os.path.join(pwd_main, 'debug','all_cond_mean_img.nii.gz'))

        with open(os.path.join(p, 'fitted_timeSeries.pkl'), 'wb') as f:
            pickle.dump(results_con, f)
        with open(os.path.join(p, 'pre_data.pkl'),) as f:
            pickle.dump(pre_data, f)
        with open(os.path.join(p, 'post_data.pkl'),) as f:
            pickle.dump(post_data, f)
        

    
        all_cond_masker = masker.fit(pre_data + post_data)
        all_cond_masker.report
        masker.report

       #with open(os.path.join(p, 'fitted_timeSeries.pkl'), 'wb') as f:
        #    pickle.dump(results_con, f)

    # --Seed masker--
    if sphere_coord != None:
        seed_masker = NiftiSpheresMasker(
            sphere_coord, radius=15, standardize=False, verbose=5
        )
        results_con["seed_pre_series"] = [ # /!!!\ Adjust this section according to report tests made up
            seed_masker.fit_transform(ts) for ts in pre_data
        ]
        results_con["seed_post_series"] = [
            seed_masker.fit_transform(ts) for ts in post_data
        ]
        # Compute seed-to-voxel correlation
        results_con["seed_to_pre_correlations"] = [
            (np.dot(brain_time_series.T, seed_time_series) / seed_time_series.shape[0])
            for brain_time_series, seed_time_series in zip(
                results_con["pre_series"], results_con["seed_pre_series"]
            )
        ]
        results_con["seed_to_post_correlations"] = [
            (np.dot(brain_time_series.T, seed_time_series) / seed_time_series.shape[0])
            for brain_time_series, seed_time_series in zip(
                results_con["post_series"], results_con["seed_post_series"]
            )
        ]

    # -- Covariance Estimation--
    covariance_measure = ConnectivityMeasure(
        kind=connectivity_measure, discard_diagonal=False,standardize = False
    )
    # Connectomes computation : returns a list of connectomes
    pre_connectomes, post_connectomes = func.compute_cov_measures(
        covariance_measure, results_con
    )
    # Connectome processing (r to Z tranf, remove neg edges, normalize)
    results_con["pre_connectomes"] = func.proc_connectomes(
        pre_connectomes,arctanh=False, remove_negw=True, normalize=False
    )
    results_con["post_connectomes"] = func.proc_connectomes(
        post_connectomes,arctanh=False, remove_negw=True, normalize=False
    )
    # weight substraction to compute change from pre to post
    results_con["diff_weight_connectomes"] = func.weight_substraction_postpre(
        results_con["post_connectomes"],
        results_con["pre_connectomes"],
    )
    # Saving
    if save_base != None:
        if os.path.exists(os.path.join(save_base, save_folder)) is False:
            os.mkdir(os.path.join(save_base, save_f/older))
        save_to = os.path.join(save_base, save_folder)

        with open(os.path.join(save_to, "dict_connectomes.pkl"), "wb") as f:
            pickle.dump(results_con, f)

    return results_con



def connectome_analyses(data, results_con, atlas_labels, save_base=None, save_folder=None):

    # Graphs metrics computation for pre and post layers : Return a dict of metrics for each subject pre.keys() = 'degree', 'betweenness', etc.
    results_graph["pre_metrics"] = graphs_regressionCV.compute_indiv_graphs_metrics(
        results_con["pre_connectomes"], data.subjects, atlas_labels
    )
    results_graph["post_metrics"] = graphs_regressionCV.compute_indiv_graphs_metrics(
        results_con["post_connectomes"], data.subjects, atlas_labels
    )
    results_graph["change_feat"] = graphs_regressionCV.metrics_diff_postpre(
         results_graph["post_metrics"], results_graph["pre_metrics"], data.subjects, exclude_keys = ['nodes', 'communities']
    )
    pre_X_weights = graphs_regressionCV.connectome2feature_matrices(
        results_con["pre_connectomes"], data.subjects
    )
    post_X_weights = graphs_regressionCV.connectome2feature_matrices(
        results_con["post_connectomes"], data.subjects
    )
    change_X_weights = post_X_weights - pre_X_weights
    print("change X shape :", change_X_weights.shape)
    Xmat_names = ["Degree", "Closeness", "Betweenness", "Clustering", "Edge_weights"]

    # Prediction of behavioral variables from connectomes
    xlsx_file = r"Hypnosis_variables_20190114_pr_jc.xlsx"
    xlsx_path = os.path.join(pwd_main, "atlases", xlsx_file)
    Y, target_columns = graphs_regressionCV.load_process_y(xlsx_path, data.subjects)
    results_graph['Y'] = Y

    # Save main results_con and prep results_dir for regression results
    with open(os.path.join(save_to, "dict_graphsMetrics.pkl"), "wb") as f:
        pickle.dump(results_graph, f)

    # CV prediciton for each connectome condition
    results_pred['behavioral'] = Y

    print("=====PRE-HYP CONNECTOMES====")
    concat = copy.deepcopy(results_graph["pre_metrics"])
    concat.update({'Edge_weights': pre_X_weights})
    results_pred["pre"] = graphs_regressionCV.regression_cv( # Manually adding the weight connectomes to concat inside function, along other graph's metric dfs
        concat,
        Y,
        target_columns,
        exclude_keys = ['nodes', 'communities'],
        rdm_seed=40,
    )
    print("=====POST-HYP CONNECTOMES====")
    concat = copy.deepcopy(results_graph["post_metrics"])
    concat.update({'Edge_weights': post_X_weights})
    results_pred["post"] = graphs_regressionCV.regression_cv(
        concat,
        Y,
        target_columns,
        exclude_keys = ['nodes', 'communities'],
        rdm_seed=40,
    )
    print("=====CHANGE-HYP CONNECTOMES====")
    concat = copy.deepcopy(results_graph["change_feat"])
    concat.update({'Edge_weights': change_X_weights}) # N x features matrix resulting from post - pre feature matrices
    results_pred["change"] = graphs_regressionCV.regression_cv(
        concat,
        Y,
        target_columns,
        exclude_keys = ['nodes', 'communities'],
        rdm_seed=40,
    )
    # Save reg results
    with open(os.path.join(save_to, "dict_regression.pkl"), "wb") as f:
        pickle.dump(results_pred, f)

    print("Saved result dict!")

    # --Prints and plot--
    if verbose:
        print([ts.shape for ts in results_con["pre_series"]])
        print([ts.shape for ts in results_con["post_series"]])
        print(atlas.shape)
        print(np.unique(atlas.get_fdata(), return_counts=True))
        for correlation_matrix in [
            results_con["pre_mean_connectome"],
            results_con["post_mean_connectome"],
            results_con["zcontrast_mean_connectome"],
        ]:  # [results_con['pre_mean_connetomes'], results_con['post_mean_connetomes']]:
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

    return results_con

