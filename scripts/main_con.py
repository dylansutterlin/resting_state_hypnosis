import sys
import argparse
import pickle
import os
import copy
import glob
import numpy as np
import pandas as pd
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
#sys.path.append(os.path.split(os.getcwd())[0]) # add '../' to path, i.e. the main path
from src import glm_func
from src import graphs_regressionCV as graphsCV
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
    fcdict : dict
        Dictionary containing all functionnal connectivity results, including timeseries, connectomes, connectome metrics, etc.
    """

    print("---LOADING DATA---")
    data = func.load_data(data_dir,pwd_main, conf_dir, n_sub=n_sub, remove_subjects=remove_subjects)
    conditions = ['baseline', 'hypnosis', 'change']
    #pre_data, post_data = prep.resample_shape_affine(data)
    pre_data = data.func_pre
    post_data = data.func_post
    fcdict = dict(subjects=data.subjects, conditions = conditions, pre_series=list(), post_series=list())
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
        fcdict['pre_series'] = load_results['pre_series']
        fcdict['post_series'] = load_results['post_series']
    else:
        fcdict["pre_series"] = [masker.fit_transform(ts) for ts in pre_masked_img]
        fcdict["post_series"] = [masker.fit_transform(ts) for ts in post_masked_img]
        fcdict['labels'] = atlas_labels
        fcdict['atlas'] = atlas
        
    print('---SEED BASED TIMESERIES EXTRACTION---')
    if sphere_coord != None:
        seed_masker = NiftiSpheresMasker(
            sphere_coord,mask_img=voxel_masker.mask_img_, radius=15, standardize=False, verbose=5
        )
        fcdict["seed_pre_series"] = [ 
            seed_masker.fit_transform(ts) for ts in pre_masked_img
        ]
        fcdict["seed_post_series"] = [
            seed_masker.fit_transform(ts) for ts in post_masked_img
        ]
        # Compute seed-to-voxel correlation
        fcdict["seed_to_pre_correlations"] = [
            (np.dot(brain_time_series.T, seed_time_series) / seed_time_series.shape[0])
            for brain_time_series, seed_time_series in zip(
                fcdict["pre_series"], fcdict["seed_pre_series"]
            )
        ]
        fcdict['seed_pre_masker'] = seed_masker
        fcdict["seed_to_post_correlations"] = [
            (np.dot(brain_time_series.T, seed_time_series) / seed_time_series.shape[0])
            for brain_time_series, seed_time_series in zip(
                fcdict["post_series"], fcdict["seed_post_series"]
            )
        ]
        fcdict['seed_post_masker'] = seed_masker # to call inverse_transform on seed_correlation
        # Changes post-pre t test on mean timseries
        res_ttest_rel = []
        tvalues = []
        for t0, t1 in zip(fcdict['seed_pre_series'], fcdict["seed_post_series"]):
            ttest_result = ttest_rel(t0, t1)
            t_statistic, p_value, degrees_of_freedom = ttest_result.statistic[0], ttest_result.pvalue[0], ttest_result.df[0]
            res_ttest_rel.append((t_statistic, p_value, degrees_of_freedom))
            tvalues.append(t_statistic)
        fcdict['ttest_rel'] = res_ttest_rel

    # -- Covariance Estimation--
    print(f'---CONNECTIVITY COMPUTATION with {connectivity_measure} estimation ---')
    connectivity_obj = ConnectivityMeasure(
        kind=connectivity_measure, discard_diagonal=False,standardize = True
    )
    # Connectomes computation : returns a list of connectomes
    pre_connectomes, post_connectomes = func.compute_cov_measures(
        connectivity_obj, fcdict
    )
    fcdict['connectivity_obj'] = connectivity_obj # to perform inverse_transform on connectomes
    # Connectome processing (r to Z tranf, remove neg edges, normalize)
   
    fcdict["pre_connectomes"] = func.proc_connectomes(
        pre_connectomes,arctanh=False, absolute_weights= False, remove_negw=False, normalize=False
    )
    fcdict["post_connectomes"] = func.proc_connectomes(
        post_connectomes,arctanh=False, absolute_weights= False, remove_negw=False, normalize=False
    )
    # weight substraction to compute change from pre to post
    if connectivity_measure == 'correlation':
        arctanh = True # arctanh (r to z transform) is used prior to change post-pre on weights
    tmp_proc_pre = func.proc_connectomes(
        fcdict["pre_connectomes"],arctanh=arctanh)
    tmp_proc_post = func.proc_connectomes(
        fcdict["post_connectomes"],arctanh=arctanh)
    fcdict["diff_weight_connectomes"] = func.weight_substraction_postpre(tmp_proc_post, tmp_proc_pre)

    # Saving
    if save_base != None:
        if os.path.exists(os.path.join(save_base, save_folder)) is False:
            os.mkdir(os.path.join(save_base, save_folder))
        save_to = os.path.join(save_base, save_folder)  
        print(f'---SAVING RESULTS to {save_to}---')

        save_to_plot = os.path.join(save_base, save_folder, 'plots')
        if os.path.exists(save_to_plot) is False:
            os.mkdir(save_to_plot)

        if os.path.exists(os.path.join(save_to, 'reports')) is False:
            os.mkdir(os.path.join(save_to, 'reports'))
        voxel_masker.generate_report().save_as_html(os.path.join(save_to, 'reports', 'voxelMasker_report.html'))
        masker.generate_report(displayed_maps='all').save_as_html(os.path.join(save_to,'reports','mapsMasker_report.html'))

        data.atlas_labels = atlas_labels
        data.save_to = save_to

        with open(os.path.join(save_to, "dict_connectomes.pkl"), "wb") as f:
            pickle.dump(fcdict, f)
        with open(os.path.join(save_to, 'data.pkl'), 'wb') as f:
            pickle.dump(data, f)
        #with open(os.path.join(save_to, 'atlas_labels.pkl'), 'wb') as f:
        #    pickle.dump(atlas_labels, f)

        # Convert connectivity matrices to txt files and save them
        # Used for comptabilities with Matlab BCT > NBS toolboxes
        if os.path.exists(os.path.join(save_to,'NBS_txtData')) is False:
            os.mkdir(os.path.join(save_to,'NBS_txtData'))
        func.export_txt_NBS(os.path.join(save_to,'NBS_txtData'), atlas, atlas_labels, fcdict['pre_connectomes'],fcdict["post_connectomes"], fcdict["diff_weight_connectomes"], data.subjects)

    print('---PREPARING AND SAVING PLOTS---')
    # Plotting connectomes and weights distribution
    for cond, matrix_list in zip(fcdict['conditions'],[
            fcdict["pre_connectomes"],fcdict['post_connectomes'], fcdict["diff_weight_connectomes"]]): 
        plot_func.dist_mean_edges(cond, matrix_list, save_to_plot)

    # Replication of Rainville et al., 2019 automaticity ~ rCBF in parietal Operculum(supramarg. gyrus in difumo64)
    Y = data.phenotype
    vd = 'Abs_diff_automaticity'
    auto = list(np.array(Y[vd])) # list convert to np.object>float (somehow?)
    mean_rCBF_diff = np.array([np.mean(post-pre) for post, pre in zip(fcdict['seed_post_series'], fcdict['seed_pre_series'])])

    plot_func.visu_correl(auto, mean_rCBF_diff, save_to_plot, vd_name = vd, vi_name = 'rCBF change in PO', title = 'Automaticity score vs Mean rCBF diff')
    # pair ttest on post-pre seed signal
    if sphere_coord != None:
        plot_func.visu_correl(auto, tvalues, save_to_plot, vd_name = vd, vi_name = 'T test change in PO', title = 'Automaticity score vs mean ttest')
    print('---DONE with connectivity matrices and plots---')

    return data, fcdict



def connectome_analyses(data, fcdict, bootstrap = 1000 ):

    subjects = data.subjects
    atlas_labels = data.atlas_labels
    save_to = data.save_to
    save_to_plots = os.path.join(save_to, 'plots')
    node_metrics = ['strength','strengthnorm', 'eigenCent','closeCent', 'betCentrality', 'degCentrality', 'clustering', 'localEfficiency']
    graphs = dict()

    print('---COMPUTING GRAPHS METRICS and CHANGE POST-PRE---')
    # Graphs metrics computation for pre and post layers : Return a dict of graphs. where keys=subjs and values=nxGraphs
    graphs['pre_graphs'], metric_ls, _ = graphsCV.compute_graphs_metrics(
        fcdict["pre_connectomes"], data.subjects, atlas_labels, out_type='dict'
    )
    graphs['post_graphs'], metric_ls, _ = graphsCV.compute_graphs_metrics(
        fcdict["post_connectomes"], data.subjects, atlas_labels, out_type='dict'
    )
    # Compute df with Nodes as rows, and metrics as columns. /subs
    graphs['pre_nodes_metrics'], keys_id = graphsCV.node_attributes2df(graphs['pre_graphs'], node_metrics)
    graphs['post_nodes_metrics'], keys_id = graphsCV.node_attributes2df(graphs['post_graphs'], node_metrics)
    # Compute change post-pre :returns list of dfs (Nodes x metrics df)
    graphs['change_nodes'] = graphsCV.node_metric_diff(graphs['post_nodes_metrics'], graphs['pre_nodes_metrics'], subjects)

    print('---RANDOMIZATION OF CONNECTOMES---')
    permNames = [f"perm_{i}" for i in range(bootstrap)]
    # Randomize connectomes : returns dict of subs with permuted connectomes (lists)
    graphs['randCon_pre'] = graphsCV.rand_conmat(fcdict['pre_connectomes'], data.subjects, n_permut=bootstrap, algo='hqs')
    graphs['randCon_post'] = graphsCV.rand_conmat(fcdict['post_connectomes'], data.subjects, n_permut=bootstrap)
    print('---checking rand matrices distribution---')
    all_pre = []                
    for sub in subjects:
        all_pre += list(graphs['randCon_pre'][sub]) # list of all permuted connectomes to mean
    plot_func.dist_mean_edges('All_rand_pre', all_pre, save_to_plots)
    all_post = []
    for sub in subjects:
        all_post += list(graphs['randCon_post'][sub])
    plot_func.dist_mean_edges('All_rand_post', all_post, save_to_plots )

    print('---COMPUTING RANDOMIZED GRAPHS---')
    # compute graphs, extracts df of nodes x metrics, and change of each subject (keys of dict)
    graphs['randCon_nodeChange_dfs'] = dict()         
    for sub in subjects: # In a loop, cause compute_graphs() is usually for a list of subjects
        subi_rand_pre, _, _ = graphsCV.compute_graphs_metrics(graphs['randCon_pre'][sub], permNames, atlas_labels, out_type='dict', verbose=f'Pre-{sub}')
        #rand_pre_graphs[sub] = subi_rand_pre  # But here the list is for one sub (list of rand mat)
        subi_rand_post,_,_ = graphsCV.compute_graphs_metrics(graphs['randCon_post'][sub], permNames, atlas_labels, out_type='dict', verbose=f'Post-{sub}')
        #rand_post_graphs[sub] = subi_rand_post
        # Graph to dataframe/n_permut
        rand_pre_dfs, ids = graphsCV.node_attributes2df(subi_rand_pre, node_metrics)
        rand_post_dfs, ids = graphsCV.node_attributes2df(subi_rand_post, node_metrics)

        # Compute diff directly on each permuted graphs_dfs/sub
        graphs['randCon_nodeChange_dfs'][sub] = graphsCV.node_metric_diff(rand_post_dfs,rand_pre_dfs, permNames)

    print('---CHANGE P VALUES---')
    # Return a df (node x metrics) of pvalues for each subjects
    graphs['pval_ls_nodes'] = graphsCV.bootstrap_pvals_df(graphs['change_nodes'], graphs['randCon_nodeChange_dfs'], subjects, mult_comp = 'fdr_bh')
    
    if os.path.exists(save_to) is False:
            os.mkdir(save_to)
    with open(os.path.join(save_to, 'graphs_dict.pkl'), 'wb') as f:
            pickle.dump(graphs, f)

    return graphs 


def prediction_analyses(data, graphs, n_permut = 5000, test_size = 0.20, pca = 0.90, verbose = False):
    # Models to test
    # - Yi ~ Strenghts, Clustering, EigenCentrality, LocalEfficiency
    # Yi ~ PO multivariate node metrics
    # Prediction of behavioral variables from connectomes
    Y = data.phenotype
    target_col = [
        "SHSS_score",
        "total_chge_pain_hypAna",
        "Chge_hypnotic_depth",
        "Mental_relax_absChange",
        "Abs_diff_automaticity",
    ]

    pred_node_metrics = ['strengthnorm', 'eigenCent'] # bet matmul error(??), 'betCentrality'] #'localEfficiency']
    #Xmat_names = ["Degree", "Closeness", "Betweenness", "Clustering", "Edge_weights"]
    cv_results = dict(pre= dict(), post=dict(), change=dict())
    cv_results['phenotype'] = Y
    single_ROI_reg = ['Supramarginal gyrus', 'Anterior Cingulate Cortex', 'Cingulate gyrus mid-anterior','Cingulate cortex posterior'] # PO, 
    subjects = data.subjects
    save_to = data.save_to
    atlas_labels = data.atlas_labels

    print(r'---CROSS-VAL REGRESSION [Yi ~ Node] METRICS---')

    # 1) Extracts metric column for each node (1 x Nodes) and stack them/sub --> (N sub x Nodes) ~ Yi
    # 2) Compute CV regression for Yi variable
    for node_metric in pred_node_metrics:
        feat_mat = pd.DataFrame(np.array([sub_mat[node_metric] for sub_mat in graphs['change_nodes']]), columns = atlas_labels, index = subjects)
        cv_results['change'][node_metric] = graphsCV.regression_cv(feat_mat, Y, target_col, pred_metric_name = node_metric, n_permut = n_permut, test_size = test_size, pca=pca) 

    for node_metric in pred_node_metrics:
        feat_mat = pd.DataFrame(np.array([sub_mat[node_metric] for sub_mat in graphs['post_nodes_metrics']]), columns = atlas_labels, index = subjects)
        cv_results['post'][node_metric] = graphsCV.regression_cv(feat_mat, Y, target_col, pred_metric_name = node_metric, n_permut = n_permut, test_size = test_size, pca=pca)

    # Edge based prediction for post and change
    post_edge_feat_mat = graphsCV.edge_attributes2df(graphs['post_graphs'], edge_metric = 'weight')
    cv_results['post']['edge_weight'] = graphsCV.regression_cv(post_edge_feat_mat, Y, target_col, pred_metric_name = 'weight', n_permut = n_permut, test_size = test_size, pca=pca)

    pre_edge_feat_mat = graphsCV.edge_attributes2df(graphs['pre_graphs'], edge_metric = 'weight')
    change_edge_feat_mat = post_edge_feat_mat - pre_edge_feat_mat
    cv_results['change']['edge_weight'] = graphsCV.regression_cv(change_edge_feat_mat, Y, target_col, pred_metric_name = 'weight', n_permut = n_permut, test_size = test_size, pca=pca)

    # ROI specific multivariate (multi-node metric) prediction
    for roi in single_ROI_reg: # compute N sub x M metrics df for prediction of Yi
        roi_feat_mat = pd.DataFrame(np.array([sub_mat.loc[roi,pred_node_metrics] for sub_mat in graphs['change_nodes']]), columns = pred_node_metrics, index = subjects) 
        cv_results['change'][roi] = graphsCV.regression_cv(roi_feat_mat, Y, target_col,  pred_metric_name = roi, n_permut= n_permut, test_size=test_size, pca=None)


    # Save reg results
    with open(os.path.join(save_to, "cv_results.pkl"), "wb") as f:
        pickle.dump(cv_results, f)

    print("Saved CV result as dict!")

    return cv_results

'''
    # --Prints and plot--
    if verbose:
        print([ts.shape for ts in fcdict["pre_series"]])
        print([ts.shape for ts in fcdict["post_series"]])
        #print(atlas.shape)
        print(np.unique(atlas.get_fdata(), return_counts=True))
        for correlation_matrix in [
            fcdict["pre_mean_connectome"],
            fcdict["post_mean_connectome"],
            fcdict["zcontrast_mean_connectome"],
        ]:  # [fcdict['pre_mean_connetomes'], fcdict['post_mean_connetomes']]:
            np.fill_diagonal(correlation_matrix, 0)
            plotting.plot_matrix(
                correlation_matrix,
                labels=atlas_labels,
                colorbar=True,
                vmax=0.8,
                vmin=-0.8,
            )
            func.plot_bilat_nodes(correlation_matrix, atlas, atlas_name)

        #plotting.plot_roi(atlas, title=atlas_name)

'''
