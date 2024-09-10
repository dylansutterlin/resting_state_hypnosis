from scripts import main_con as m
import os
import pickle

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


p = r"/data/rainville/HYPNOSIS_ASL_ANALYSIS/CBF_normalized"
conf_dir = (
    r"/data/rainville/HYPNOSIS_ASL_ANALYSIS")
pwd_main = r"/data/rainville/dSutterlin/projects/resting_hypnosis/resting_state_hypnosis"

#p = r'/home/dsutterlin/projects/test_data/ASL_RS_hypnosis/CBF_4D_normalized'
#conf_dir = False
#pwd_main = r"/home/dsutterlin/projects/resting_state_hypnosis/resting_state_hypnosis"

p = p
pwd_main=pwd_main,
conf_dir=conf_dir,
save_base= os.path.join(pwd_main, 'debug'),
save_folder="yeo17_testing",
atlas_name="yeo_17",
sphere_coord = [(54, -28, 26)],
connectivity_measure="correlation",
n_sub = None,
verbose=True,
remove_ROI_maps = [8,14,43], # based on masker report, outside brain or no interest
remove_subjects = ['APM_07_H1', 'APM_11_H1', 'APM_22_H2'] # based on rainville et al., 2019 and .xlsx file


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


import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from nilearn import plotting
n_roi = np.unique(a.get_fdata(), return_counts=True)[0].max()
colors = plt.cm.viridis(np.linspace(0, 1, n_roi))  # You can change 'viridis' to any other colormap
custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', colors, N=n_roi)

plotting.plot_roi(atlas, title=atlas_name, cmap=custom_cmap, colorbar=True)
plt.savefig(os.path.join(save_folder, 'atlas_view.png'))


print('---TIMESERIES EXTRACTION FOR ROIs---') 
fcdict["pre_series"] = [masker.fit_transform(ts) for ts in pre_masked_img]
fcdict["post_series"] = [masker.fit_transform(ts) for ts in post_masked_img]
fcdict['labels'] = atlas_labels
fcdict['atlas'] = atlas


print('---WHOLE BRAIN MASKING ON TIMESERIES---')
#p_data = r'projects/test_data/ASL_RS_hypnosis/CBF_4D_normalized'
voxel_masker = NiftiMasker(
    mask_strategy="whole-brain-template",
    standardize = False,
    high_variance_confounds = False,
    verbose=5,
)  

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
    masker.generate_report().save_as_html(os.path.join(save_to,'reports','labelsMasker_report.html'))

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



'''
graphs = m.connectome_analyses(data, fcdict, bootstrap = 400)

with open(os.path.join(pwd_main,'debug', 'difumo64_correl_noProc', 'data.pkl'),'rb') as f:
    data = pickle.load(f)

with open(os.path.join(pwd_main,'debug', 'difumo64_correl_noProc', 'graphs_dict.pkl'),'rb') as f:
    graphs = pickle.load(f)

cv_results = m.prediction_analyses(data, graphs, n_permut = 200, verbose = False)

import matplotlib.pyplot as plt
import pickle
with open(os.path.join(pwd_main, 'debug', 'difumo64_correlation', 'dict_connectomes.pkl'), 'rb') as f:
    results_con = pickle.load(f)
with open(os.path.join(pwd_main, 'debug', 'difumo64_correlation', 'data.pkl'), 'rb') as f:
    data = pickle.load(f)
with open(os.path.join(pwd_main, 'debug', 'difumo64_correlation', 'atlas_labels.pkl'), 'rb') as f:
    labels = pickle.load(f)
save_to = os.path.join(pwd_main, 'debug', 'difumo64_correlation')
'''






