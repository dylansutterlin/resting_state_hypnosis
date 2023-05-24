
import argparse

import os
import glob
import numpy as np
import nibabel as nib
from nilearn.maskers import NiftiMapsMasker, NiftiLabelsMasker, MultiNiftiLabelsMasker, MultiNiftiMapsMasker
from nilearn.connectome import ConnectivityMeasure
import func
from nilearn import datasets, plotting
from nilearn.regions import connected_label_regions


def con_matrix(data_dir, save_to, path_to_atlas = None, atlas_type = 'labels', connectivity_measure = 'correlation', con_stat = 'diff', plot_atlas = False, verbose = True):
    """
    Computes connectivity matrices for many subjects.


    """


    # --Data--
    data = func.load_data(data_dir)
    idcs = [i + 1 for i in range(0,len(data.subjects))]
    conditions = ['pre_hyp', 'post_hyp', 'diff']
    pre_data = data.pre_hyp
    post_data = data.post_hyp
    print(pre_data)
    results = dict(pre_series = list(), post_series =  list())
    Y = data.phenotype['PercDiffAutomat'][2:-2].T[1]

    # --Atlas choices--
    #atlas = datasets.fetch_atlas_msdl()
    atlas = datasets.fetch_atlas_basc_multiscale_2015(version="sym", resolution=64) #, resolution=None)
    #dataset = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
    atlas_filename = datasets.fetch_atlas_yeo_2011()['thick_7']
    atlas_name = 'yeo_7'
    atlas_labels  = ['Visual', 'Somatosensory', 'Dorsal Attention', 'Ventral Attention', 'Limbic', 'Frontoparietal', 'Default']
    print(nib.load(atlas_filename).shape)
    from nilearn import plotting
    plotting.plot_roi(atlas_filename, title=atlas_name)
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
    masker.fit(pre_data)
    results['pre_series'] = [masker.transform(ts) for ts in pre_data] # Confounds?
    masker.fit(post_data)
    results['post_series'] =  [masker.transform(ts) for ts in post_data] # A list of time series for each sub
    print(results['post_series'][0].shape)
    # -- Covariance Estimation--
    correlation_measure = ConnectivityMeasure(kind=connectivity_measure,  discard_diagonal=True)
    results['pre_connectomes'] = correlation_measure.fit_transform(results['pre_series'])
    results['pre_mean_connetomes'] = correlation_measure.mean_
    results['post_connectomes'] =  correlation_measure.fit_transform(results['post_series'])
    results['post_mean_connetomes'] = correlation_measure.mean_
    #results['diff_mean_connectomes'] =  func.diff_rz(results['pre_connectomes'], results['post_connectomes'])
    results['diff_connectomes'] = [post - pre for post, pre in zip(results['post_mean_connetomes'],results['pre_mean_connetomes'])]
    print(results['pre_connectomes'][0].shape, len(results['pre_connectomes']))
    # --Save--
    for idx, sub in enumerate(data.subjects):
        np.save(os.path.join(save_to, f'{sub}_{conditions[0]}_connectomes'), results['pre_connectomes'][idx], allow_pickle=True)
        np.save(os.path.join(save_to, f'{sub}_{conditions[1]}_connectomes'), results['post_connectomes'][idx], allow_pickle=True)
        np.save(os.path.join(save_to, f'{sub}_{conditions[2]}_connectomes'), results['diff_connectomes'][idx], allow_pickle=True)


    # --Stats--
    features = np.asarray([np.load(os.path.join(save_to, f'{sub}_{conditions[2]}_connectomes.npy'), allow_pickle=True) for sub in data.subjects])
    Y = data.phenotype['PercDiffAutomat']
    #! pheno_df = data.phenotypic
    # Best way to format data?

    if verbose:
        # Mask out the major diagonal
        for correlation_matrix in [results['pre_connectomes'][0], results['post_connectomes'][0], results['diff_connectomes']]:#[results['pre_mean_connetomes'], results['post_mean_connetomes']]:
            np.fill_diagonal(correlation_matrix, 0)
            print(correlation_matrix.shape)
            print(atlas_labels)
            plotting.plot_matrix(
                correlation_matrix, labels=atlas_labels, colorbar=True, vmax=0.8, vmin=-0.8
            )

            coordinates = plotting.find_parcellation_cut_coords(labels_img=atlas_filename)
            plotting.plot_connectome(
                correlation_matrix,
                coordinates,
                edge_threshold=None,
                title= atlas_name,
            )
            plotting.show()



'''
# plot connectivity matrix

    matrix = np.load(os.path.join('data/derivatives/connectomes', os.listdir('data/derivatives/connectomes')[0]))
    plotting.plot_matrix(squareform(matrix), vmin = -1, vmax = 1, labels=masker.labels_)
    plt.savefig('results/plots/connectivity_matrix.svg', format='svg')
    plt.clf()
'''

p = r'E:\Users\Dylan\Desktop\UdeM_H22\E_PSY3008\data_desmartaux\test_data_ASL'
save_to = r'C:\Users\Dylan\Desktop\UM_Bsc_neurocog\E22\Projet_Ivado_rainvillelab\results\test_connec'
con_matrix(p, save_to, path_to_atlas = datasets.fetch_atlas_basc_multiscale_2015())
