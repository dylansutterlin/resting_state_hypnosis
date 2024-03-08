from scripts import main_con as m
import os


#p = r"/data/rainville/HYPNOSIS_ASL_ANALYSIS/CBF_normalized"
#conf_dir = (
#    r"/data/rainville/HYPNOSIS_ASL_ANALYSIS")
#pwd_main = r"/data/rainville/dSutterlin/projects/resting_hypnosis/resting_state_hypnosis"

p = r'/home/dsutterlin/projects/test_data/ASL_RS_hypnosis/CBF_4D_normalized'
conf_dir = False
pwd_main = r"/home/dsutterlin/projects/resting_state_hypnosis/resting_state_hypnosis"

data, fcdict, atlas_labels, save_to = m.con_matrix(
    p,
    pwd_main=pwd_main,
    conf_dir=conf_dir,
    save_base= os.path.join(pwd_main, 'debug'),
    save_folder="difumo64_correl_negw",
    atlas_name="difumo64",
    sphere_coord = [(54, -28, 26)],
    connectivity_measure="correlation",
    n_sub = 3,
    verbose=True,
    remove_ROI_maps = [8,14,43], # based on masker report, outside brain or no interest
    remove_subjects = ['APM_07_H1', 'APM_11_H1', 'APM_22_H2'] # based on rainville et al., 2019 and .xlsx file
    )

graphs = m.connectome_analyses(data, fcdict, n_iter = 100)

cv_results = m.prediction_analyses(data, graphs, save_to, n_permut = 100, verbose = False)

'''
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






