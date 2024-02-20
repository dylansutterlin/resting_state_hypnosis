from scripts import main_con as m
import os

def main():
    
    r"""
    p = r"/data/rainville/HYPNOSIS_ASL_ANALYSIS/CBF_normalized"
    conf_dir = (
        r"/data/rainville/HYPNOSIS_ASL_ANALYSIS")
    pwd_repo = r"/data/rainville/dSutterlin/projects/resting_hypnosis/resting_state_hypnosis"
    #p = r"/home/p1226014/projects/def-rainvilp/p1226014/data/rainville2019_ASL_preproc/CBF_normalized"
    #conf_dir = (r"/home/p1226014/projects/def-rainvilp/p1226014/data/rainville2019_ASL_preproc")
    #pwd_repo = r"/home/p1226014/projects/def-rainvilp/p1226014/hypnosis_rest"

    # p = r"/data/rainville//HYPNOSIS_ASL_ANALYSIS/CBF_normalized/"
    # else_dir = r"/data/rainville/HYPNOSIS_ASL_ANALYSIS"
    # save_base = r"/data/rainville/dylanSutterlin/results/connectivity"
    # atlas_name = ["yeo_7", "difumo64"]


    # )
    # save_base = r"C:\Users\Dylan\Desktop\UM_Bsc_neurocog\E22\Projet_Ivado_rainvillelab\results\results_con"
    # atlas_name = ["yeo_7", "difumo64"]
    print('==========TEST MODIF MAIN : CORRELATION UPDATE LATEST METHOD==========')
    main.con_matrix(
        p,
        conf_dir=conf_dir,
        pwd_main=os.path.join(pwd_repo, 'pipeline'),
        save_base= os.path.join(pwd_repo, 'results'),
        save_folder="difumo64_tangent_neg0_SVR",
        atlas_name="difumo64",
        sphere_coord = [
            (54, -28, 26),
            (-20, -26, -14),
            (-2, 20, 32),
            (-8, 44, 28),
            (-6, -26, 46),
        ],
        
        connectivity_measure="tangent",
        n_sub = None,
        verbose=False,
    )


    """

    a = 'trash'
    print(a)
    return a

p = r"/data/rainville/HYPNOSIS_ASL_ANALYSIS/CBF_normalized"
conf_dir = (
    r"/data/rainville/HYPNOSIS_ASL_ANALYSIS")
pwd_main = r"/data/rainville/dSutterlin/projects/resting_hypnosis/resting_state_hypnosis"

#p = r'/home/dsutterlin/projects/test_data/ASL_RS_hypnosis/CBF_4D_normalized'
#conf_dir = False
#pwd_main = r"/home/dsutterlin/projects/resting_state_hypnosis/resting_state_hypnosis"

m.con_matrix(
    p,
    pwd_main=pwd_main,
    conf_dir=conf_dir,
    save_base= os.path.join(pwd_main, 'debug'),
    save_folder="difumo64_correlation",
    atlas_name="difumo64",
    sphere_coord = [(54, -28, 26)],
    connectivity_measure="correlation",
    n_sub = None,
    verbose=True
    )

import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

with open(os.path.join(pwd_main, 'debug', 'difumo64_correlation', 'dict_connectomes.pkl'), 'rb') as f:
    results_con = pickle.load(f)
with open(os.path.join(pwd_main, 'debug', 'difumo64_correlation', 'data.pkl'), 'rb') as f:
    data = pickle.load(f)
with open(os.path.join(pwd_main, 'debug', 'difumo64_correlation', 'atlas_labels.pkl'), 'rb') as f:
    labels = pickle.load(f)
save_to = os.path.join(pwd_main, 'debug', 'difumo64_correlation')

#m.connectome_analyses(data, results_con, labels)




