import os
import glob as glob
import pandas as pd
import numpy as np
from sklearn.utils import Bunch

def load_data(path):
    """
    Load subject information into memory

    """
    data = Bunch(

        subjects = [sub for sub in os.listdir(path) if 'APM' in sub],
        pre_hyp = [glob.glob(os.path.join(path, sub, '*before*', '*4D*'))[0] for sub in os.listdir(path) if 'APM' in sub],
        post_hyp = [glob.glob(os.path.join(path, sub, '*during*', '*4D*'))[0] for sub in os.listdir(path) if 'APM' in sub],
        anat = [glob.glob(os.path.join(path, sub, 'anatomy', '*.nii'))[0] for sub in os.listdir(path) if 'APM' in sub],
        phenotype = pd.DataFrame(pd.read_excel(glob.glob(os.path.join(path, '*variables*'))[0], sheet_name = 0, index_col = 1, header = 2))
        )
    
    return data

def diff_rz(pre_connectome, post_connectome, verbose = True):
    '''

    pre (list) : list with all connectomes from pre condition
    post(list) : List with all connectomes from post condition

    '''

    diff = list()
    for pre, post in zip(pre_connectome, post_connectome):
        res  = np.arctanh(pre) - np.arctanh(post)
        diff.append(res)

    if verbose:
        print('Computing diff in lists of {}, {} connectome with r to Z arctanh func./n Diff matrix has shape : {} '.format(len(pre_connectome), len(post_connectome), diff[0].shape))

    return diff

#def reg_model(conne)
