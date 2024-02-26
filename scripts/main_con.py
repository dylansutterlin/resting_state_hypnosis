import sys
import argparse
import pickle
import os
import copy
import glob
import numpy as np
import nibabel as nib
from scipy.stats import pearsonr
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
from scripts import func, plot_func
from src import masker_preprocessing as prep

from scipy.stats import ttest_rel 
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
    remove_ROI_maps = False,
    remove_subjects = False
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
    
    remove_ROI : list, optional
        List of ROI to remove from the atlas, by default [43], to remove maps 43 (paracentral lobule superior) of DiFuMo64

    remove_subjects : list, optional
        List of subjects to remove from the data, by default False. Passed in func.load_data()
    Returns
    -------
    results_con : dict
        Dictionary containing all connectivity results, including connectomes, connectome metrics, etc.
    """

    print("---LOADING DATA---")
    data = func.load_data(data_dir,pwd_main, conf_dir, n_sub=n_sub, remove_subjects=remove_subjects)
    conditions = ['baseline', 'hypnosis', 'change']
    #pre_data, post_data = prep.resample_shape_affine(data)
    pre_data = data.func_pre
    post_data = data.func_post
    results_con = dict(subjects=data.subjects, conditions = conditions, pre_series=list(), post_series=list())
    results_graph = dict(pre=dict(), post=dict(), change=dict())
    results_pred = dict(pre=dict(), post=dict(), change=dict())
    
    print('----ATLAS SELECTION---')
    if atlas_name != None:
        atlas, atlas_labels, atlas_type, confounds = func.load_choose_atlas(
            pwd_main, atlas_name, remove_ROI_maps = remove_ROI_maps, bilat=True
        ) # confounds is not used further

        
    print('---WHOLE BRAIN MASKING ON TIMESERIES---')
    #p_data = r'projects/test_data/ASL_RS_hypnosis/CBF_4D_normalized'
    voxel_masker = NiftiMasker(
        mask_strategy="whole-brain-template",
        standardize = False,
        high_variance_confounds = False,
        verbose=5,
    )
    #voxel_masker.fit(pre_data + post_data)
    pre_masked_img = []
    post_masked_img = []
    for i in range(len(pre_data)): # inverse transform to remove contour mask on data
        tmp_img = voxel_masker.fit_transform(pre_data[i])
        pre_masked_img.append(voxel_masker.inverse_transform(tmp_img))
    for i in range(len(post_data)):
        tmp_img = voxel_masker.fit_transform(post_data[i])
        post_masked_img.append(voxel_masker.inverse_transform(tmp_img))

    print('---ATLAS MASK FITTING---')
    # Return NiftiMaker object yet to be fitted
    if atlas_name == None:
        masker = voxel_masker
    else:
        masker = prep.choose_atlas_masker(
            atlas = atlas, atlas_type =atlas_type, mask_img=voxel_masker.mask_img_, resampling_target= 'data', standardize='zscore_sample', verbose=5
        ) 

    print('---TIMESERIES EXTRACTION FOR ROIs---') 
    p = os.path.join(pwd_main, 'debug')
    input_timeseries=False
    if input_timeseries != False:
        with open(os.path.join(p,'fitted_timeSeries.pkl'), 'rb') as f:
            load_results = pickle.load(f)
        results_con['pre_series'] = load_results['pre_series']
        results_con['post_series'] = load_results['post_series']
    else:
        results_con["pre_series"] = [masker.fit_transform(ts) for ts in pre_masked_img]
        results_con["post_series"] = [masker.fit_transform(ts) for ts in post_masked_img]
        results_con['labels'] = atlas_labels
        results_con['atlas'] = atlas
        
    print('---SEED BASED TIMESERIES EXTRACTION---')
    if sphere_coord != None:
        seed_masker = NiftiSpheresMasker(
            sphere_coord,mask_img=voxel_masker.mask_img_, radius=15, standardize=False, verbose=5
        )
        results_con["seed_pre_series"] = [ # /!!!\ Adjust this section according to report tests made up
            seed_masker.fit_transform(ts) for ts in pre_masked_img
        ]
        results_con["seed_post_series"] = [
            seed_masker.fit_transform(ts) for ts in post_masked_img
        ]
        # Compute seed-to-voxel correlation
        results_con["seed_to_pre_correlations"] = [
            (np.dot(brain_time_series.T, seed_time_series) / seed_time_series.shape[0])
            for brain_time_series, seed_time_series in zip(
                results_con["pre_series"], results_con["seed_pre_series"]
            )
        ]
        results_con['seed_pre_masker'] = seed_masker
        results_con["seed_to_post_correlations"] = [
            (np.dot(brain_time_series.T, seed_time_series) / seed_time_series.shape[0])
            for brain_time_series, seed_time_series in zip(
                results_con["post_series"], results_con["seed_post_series"]
            )
        ]
        results_con['seed_post_masker'] = seed_masker # to call inverse_transform on seed_correlation
        # Changes post-pre t test on mean timseries
        res_ttest_rel = []
        tvalues = []
        for t0, t1 in zip(results_con['seed_pre_series'], results_con["seed_post_series"]):
            ttest_result = ttest_rel(t0, t1)
            t_statistic, p_value, degrees_of_freedom = ttest_result.statistic[0], ttest_result.pvalue[0], ttest_result.df[0]
            res_ttest_rel.append((t_statistic, p_value, degrees_of_freedom))
            tvalues.append(t_statistic)
        results_con['ttest_rel'] = res_ttest_rel

    # -- Covariance Estimation--
    print(f'---CONNECTIVITY COMPUTATION with {connectivity_measure} estimation ---')
    connectivity_obj = ConnectivityMeasure(
        kind=connectivity_measure, discard_diagonal=False,standardize = True
    )
    # Connectomes computation : returns a list of connectomes
    pre_connectomes, post_connectomes = func.compute_cov_measures(
        connectivity_obj, results_con
    )
    results_con['connectivity_obj'] = connectivity_obj # to perform inverse_transform on connectomes
    # Connectome processing (r to Z tranf, remove neg edges, normalize)
   
    results_con["pre_connectomes"] = func.proc_connectomes(
        pre_connectomes,arctanh=True, absolute_weights= True, remove_negw=False, normalize=False
    )
    results_con["post_connectomes"] = func.proc_connectomes(
        post_connectomes,arctanh=True, absolute_weights= True, remove_negw=False, normalize=False
    )
    # weight substraction to compute change from pre to post
    results_con["diff_weight_connectomes"] = func.weight_substraction_postpre(
        results_con["post_connectomes"],
        results_con["pre_connectomes"],
    )
    # Saving
    if save_base != None:
        if os.path.exists(os.path.join(save_base, save_folder)) is False:
            os.mkdir(os.path.join(save_base, save_folder))
        save_to = os.path.join(save_base, save_folder)
        print(f'---SAVING RESULTS to {save_to}---')

        if os.path.exists(os.path.join(save_to, 'reports')) is False:
            os.mkdir(os.path.join(save_to, 'reports'))
        voxel_masker.generate_report().save_as_html(os.path.join(save_to, 'reports', 'voxelMasker_report.html'))
        masker.generate_report(displayed_maps='all').save_as_html(os.path.join(save_to,'reports','mapsMasker_report.html'))

        with open(os.path.join(save_to, "dict_connectomes.pkl"), "wb") as f:
            pickle.dump(results_con, f)
        with open(os.path.join(save_to, 'data.pkl'), 'wb') as f:
            pickle.dump(data, f)
        with open(os.path.join(save_to, 'atlas_labels.pkl'), 'wb') as f:
            pickle.dump(atlas_labels, f)

        # Convert connectivity matrices to txt files and save them
        # Used for comptabilities with Matlab BCT > NBS toolboxes
        if os.path.exists(os.path.join(save_to,'NBS_txtData')) is False:
            os.mkdir(os.path.join(save_to,'NBS_txtData'))
        func.export_txt_NBS(os.path.join(save_to,'NBS_txtData'), atlas, atlas_labels, results_con['pre_connectomes'],results_con["post_connectomes"] ,data.subjects)
    
    print('---PREPARING AND SAVING PLOTS---')
    # Plotting connectomes and weights distribution
    for cond, matrix_list in zip(results_con['conditions'],[
            results_con["pre_connectomes"],results_con['post_connectomes'], results_con["diff_weight_connectomes"]]): 
        plot_func.dist_mean_edges(cond, matrix_list, save_to)
        
    # Replication of Rainville et al., 2019 automaticity ~ rCBF in parietal Operculum(supramarg. gyrus in difumo64)
    Y = data.phenotype
    vd = 'Abs_diff_automaticity'
    auto = list(np.array(Y[vd])) # list convert to np.object>float (somehow?)
    mean_rCBF_diff = np.array([np.mean(post-pre) for post, pre in zip(results_con['seed_post_series'], results_con['seed_pre_series'])])

    plot_func.visu_correl(auto, mean_rCBF_diff, save_to, vd_name = vd, vi_name = 'rCBF change in PO', title = 'Automaticity score vs Mean rCBF diff')
    # pair ttest on post-pre seed signal
    if sphere_coord != None:
        plot_func.visu_correl(auto, tvalues, save_to, vd_name = vd, vi_name = 'T test change in PO', title = 'Automaticity score vs mean ttest')
    print('---DONE with connectivity matrices and plots---')


    return data, results_con, atlas_labels



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
    
    # Permutation test on graph metrics
    rand_graphs = dict()
    
       
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

