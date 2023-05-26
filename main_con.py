
import argparse

import os
import glob
import numpy as np
import nibabel as nib
from nilearn.maskers import NiftiMapsMasker, NiftiLabelsMasker, MultiNiftiLabelsMasker, MultiNiftiMapsMasker
from nilearn.connectome import ConnectivityMeasure
import func
from nilearn import datasets, plotting, image
from nilearn.regions import connected_label_regions


def con_matrix(data_dir, save_to, path_to_atlas = None, atlas_type = 'labels', connectivity_measure = 'correlation', plot_atlas = False, verbose = False):
    """_summary_

    Parameters
    ----------
    data_dir : str
        Path to data
    save_to : str
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
    conditions = ['pre_hyp', 'post_hyp', 'contrast']
    pre_data = data.pre_hyp
    post_data = data.post_hyp
    results = dict(pre_series = list(), post_series =  list())
    Y = data.phenotype['PercDiffAutomat'][2:-2].T[1]

    # --Atlas choices--
    atlas, atlas_filename, atlas_labels = load_choose_atlas('')
    #atlas = datasets.fetch_atlas_msdl()
    atlas = datasets.fetch_atlas_basc_multiscale_2015(version="sym", resolution=64) #, resolution=None)
    #dataset = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
    atlas_filename = datasets.fetch_atlas_yeo_2011()['thick_17']
    atlas_name = 'yeo_17'
    atlas_labels  = ['Visual', 'Somatosensory', 'Dorsal Attention', 'Ventral Attention', 'Limbic', 'Frontoparietal', 'Default']

    if atlas_name == 'yeo_17':

        # Missing ROIs correction
        load_mask = nib.load(atlas_filename)
        filt_mask = np.array(load_mask.dataobj)
        filt_mask[filt_mask == 9.] = 0 # 9. is the label of this ROI we are removing

        # making ROIs bilateral
        mask_data_right[mask_data_right>0] += mask_data.max()
        bilateral_mask_data = mask_data_left + mask_data_right

        atlas_labels = [str(int(x)) for x in np.unique(filt_mask)][1:] # Remove 1st as it is the background
        atlas_filename = image.new_img_like(load_mask, filt_mask)


    # --Labels--
    #region_labels = connected_label_regions(atlas)

    #--Masker parameters--
    if atlas_type == 'maps':
         masker = NiftiMapsMasker(
         maps_img=atlas_filename,
         mask_imgs = atlas_labels,
         standardize= 'zscore_sample',
         memory="nilearn_cache",
         verbose=5,
         )
    elif atlas_type == 'labels':
        #labels = atlas.labels
        masker = NiftiLabelsMasker(
            labels_img=atlas_filename,
            labels = atlas_labels,
            standardize='zscore_sample',
            resampling_target = 'data')
        print(' Labeled masker!')

    for img in pre_data:
        load = nib.load(img)
        print(load.shape)

   # --Fit and apply mask--
    #conf, sample_mask = load_confounds_strategy(func_path, denoise_strategy = 'simple', motion = 'basic', global_signal = 'basic')
    #masker.fit(pre_data)
    results['pre_series'] = [masker.fit_transform(ts) for ts in pre_data] # Confounds?
    #masker.fit(post_data)
    results['post_series'] =  [masker.fit_transform(ts) for ts in post_data] # A list of time series for each sub
    print([ts.shape for ts in results['pre_series']])
    print([ts.shape for ts in results['post_series']])

    # Filter TS !!!
    #for i, shapes, sub in enumerate()
    #idcs = [idcs for idcs, index in enumerate(data.phenotype.index) if index in rename_sub] # Check indices of Y[i] of sub included in analysis
    #y_auto = np.array(y_full_auto[idcs])

    # -- Covariance Estimation--
    correlation_measure = ConnectivityMeasure(kind=connectivity_measure,  discard_diagonal=True)

    results['pre_connectomes'] = correlation_measure.fit_transform(results['pre_series'])
    results['pre_mean_connectome'] = correlation_measure.mean_
    results['post_connectomes'] =  correlation_measure.fit_transform(results['post_series'])
    results['post_mean_connectome'] = correlation_measure.mean_
    results['contrast_mean_connectome'] =  results['post_mean_connectome'] - results['pre_mean_connectome']
    results['contrast_connectomes'] = [post - pre for post, pre in zip(results['post_connectomes'],results['pre_connectomes'])]

    # --Save--
    for idx, sub in enumerate(data.subjects):
        np.save(os.path.join(save_to, f'{sub}_{conditions[0]}_connectomes'), results['pre_connectomes'][idx], allow_pickle=True)
        np.save(os.path.join(save_to, f'{sub}_{conditions[1]}_connectomes'), results['post_connectomes'][idx], allow_pickle=True)
        np.save(os.path.join(save_to, f'{sub}_{conditions[2]}_connectomes'), results['contrast_connectomes'][idx], allow_pickle=True)
    np.save(os.path.join(save_to, f'{conditions[0]}_mean_connectome'), results['pre_mean_connectome'], allow_pickle=True)
    np.save(os.path.join(save_to, f'{conditions[1]}_mean_connectome'), results['post_mean_connectome'], allow_pickle=True)
    np.save(os.path.join(save_to, f'{conditions[2]}_mean_connectome'), results['contrast_mean_connectome'], allow_pickle=True)


    # --Stats--
    matrices = np.asarray([np.load(os.path.join(save_to, f'{sub}_{conditions[2]}_connectomes.npy'), allow_pickle=True) for sub in data.subjects])
    y_full_auto = data.phenotype['Unnamed: 68'] # abs. diff. in perceived automaticity
    # Access selected sub based on id in y
    rename_sub = [f'APM{num}' for num in [sub[4:6] for sub in data.subjects]] # Will rename 'APM_01_H1' with 'APM01'
    idcs = [idcs for idcs, index in enumerate(data.phenotype.index) if index in rename_sub] # Check indices of Y[i] of sub included in analysis
    y_auto = np.array(y_full_auto[idcs])

    # --X/features (vectorize each connectome)--
    tril_mask = np.tril(np.ones(results['pre_connectomes'].shape[-2:]), k=-1).astype(bool) # ! assuming same shape for other matrices (post, and contrast)
    results['preX'] = np.stack([results['pre_connectomes'][i][..., tril_mask] for i in range(0,len(results['pre_connectomes']))], axis = 0)
    results['postX'] = np.stack([results['post_connectomes'][i][..., tril_mask] for i in range(0,len(results['post_connectomes']))], axis = 0)
    results['contrastX'] = np.stack([results['contrast_connectomes'][i][..., tril_mask] for i in range(0,len(results['contrast_connectomes']))], axis = 0)

    np.save(os.path.join(save_to, f'features_pre'), results['preX'], allow_pickle=True)
    np.save(os.path.join(save_to, f'features_post'), results['postX'], allow_pickle=True)
    np.save(os.path.join(save_to, f'features_contrast'), results['contrastX'], allow_pickle=True)
    np.save(os.path.join(save_to, f'Y'), y_auto, allow_pickle=True)


    # --Prints and plot--
    if verbose:

        for correlation_matrix in [results['pre_mean_connectome'], results['post_mean_connectome'], results['contrast_mean_connectome']]:#[results['pre_mean_connetomes'], results['post_mean_connetomes']]:
            np.fill_diagonal(correlation_matrix, 0)
            plotting.plot_matrix(correlation_matrix, labels=atlas_labels, colorbar=True, vmax=0.8, vmin=-0.8)
            func.plot_bilat_nodes(correlation_matrix, atlas_filename, atlas_name)

        plotting.plot_roi(atlas_filename, title=atlas_name)
'''
# plot connectivity matrix

    matrix = np.load(os.path.join('data/derivatives/connectomes', os.listdir('data/derivatives/connectomes')[0]))
    plotting.plot_matrix(squareform(matrix), vmin = -1, vmax = 1, labels=masker.labels_)
    plt.savefig('results/plots/connectivity_matrix.svg', format='svg')
    plt.clf()
'''

p = r'E:\Users\Dylan\Desktop\UdeM_H22\E_PSY3008\data_desmartaux\HYPNOSIS_ASL_DATA'
save_to = r'C:\Users\Dylan\Desktop\UM_Bsc_neurocog\E22\Projet_Ivado_rainvillelab\results\partial_connect_hyp_yeo17thick'
con_matrix(p, save_to, path_to_atlas = datasets.fetch_atlas_basc_multiscale_2015(), connectivity_measure = 'partial correlation')
