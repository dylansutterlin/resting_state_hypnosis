import os
import glob as glob
import pandas as pd
import numpy as np
import nibabel as nib
from sklearn.utils import Bunch
from nilearn import plotting
from nilearn.image import new_img_like, load_img
from nilearn import datasets, image
from matplotlib import cm
from nilearn.plotting import plot_glass_brain
from nilearn.plotting import find_probabilistic_atlas_cut_coords
from sklearn.covariance import GraphicalLassoCV
from sklearn.preprocessing import MinMaxScaler
from nilearn.connectome import GroupSparseCovarianceCV


def load_data(path, main_cwd, conf_path = False, n_sub = None):
    """
    Load subject information into memory

    Parameters
    ----------
    path : str
        Path to fmri data organized in subfolders for each subject
    main_cwd : str
        Path to folder where main.py is located. Used to access support files like phenotype files and atlases
    conf_path : str
        Path to support files like masks of data and confounds
    n_sub : int
        Used to limit the number of subjects to be loaded
        If None, all subjects will be loaded
    Returns
    -------
    data : Bunch

    """

    sorted_subs = sorted([sub for sub in os.listdir(path) if "APM" in sub])
    if n_sub!= None:
        sorted_subs = sorted_subs[:n_sub]

    data = Bunch(
        subjects=sorted_subs,
        func_pre=[
            glob.glob(os.path.join(path, sub, "wcbf_0_srASL_4D_before_4D.nii"))[0]
            for sub in sorted_subs
        ],
        func_post=[
            glob.glob(os.path.join(path, sub, "*wcbf_0_srASL_4D_during_4D.nii"))[0]
            for sub in sorted_subs
        ],
        pre_masks=[
            glob.glob(
                os.path.join(
                    conf_path, sub, "*before*", "mskwmeanCBF_0_srASL_4D_before*"
                )
            )[0]
            for sub in sorted_subs
        ],
        post_masks=[
            glob.glob(
                os.path.join(
                    conf_path, sub, "*during*", "mskwmeanCBF_0_srASL_4D_during*"
                )
            )[0]
            for sub in sorted_subs
        ],
        anat=[
            glob.glob(os.path.join(else_path, sub, "*MEMPRAGE", "wms*.nii"))[0]
            for sub in sorted_subs
        ],
        phenotype=pd.DataFrame(
            pd.read_excel(
                glob.glob(os.path.join(main_cwd, "atlases", "*variables*"))[0],
                sheet_name=0,
                index_col=1,
                header=2,
            )
        ),
    )
    
    if conf_path:
        data.confounds_pre_hyp = [
            pd.read_csv(file, sep="\s+", header=None, names=["1", "2", "3", "4"])
            for file in [
                glob.glob(os.path.join(conf_path, sub, "*before*", "globalsg_0.txt"))[0]
                for sub in sorted_subs
            ]
        ]
        data.confounds_post_hyp = [
            pd.read_csv(file, sep="\s+", header=None, names=["1", "2", "3", "4"])
            for file in [
                glob.glob(os.path.join(conf_path, sub, "*during*", "globalsg_0.txt"))[0]
                for sub in sorted_subs
            ]
        ]

        if conf_path:
            data.confounds_pre_hyp = [
                pd.read_csv(file, sep="\s+", header=None, names=["1", "2", "3", "4"])
                for file in [
                    glob.glob(os.path.join(conf_path, sub, "*before*", "globalsg_0.txt"))[0]
                    for sub in sorted_subs
                ]
            ]
            data.confounds_post_hyp = [
                pd.read_csv(file, sep="\s+", header=None, names=["1", "2", "3", "4"])
                for file in [
                    glob.glob(os.path.join(conf_path, sub, "*during*", "globalsg_0.txt"))[0]
                    for sub in sorted_subs
                ]
            ]
            
    return data


def load_choose_atlas(main_cwd, atlas_name, mask_conf = False, bilat=True):
    '''
    Loads the chosen atlas into memory
    Parameters
    ----------
    main_cwd : str
        Pipeline directory where main.py is located
    atlas_name : str
        Atlas to be loaded. Choices are: "yeo_7", "yeo_17" and "difumo64". BASC is not tested
    mask_conf : Bool.
            *Not updated, could be used with DiaFuMo64. Set to 0 in meantime.
    bilat : bool, optional
        Used in for the yeo case where ROIs are unilateral. Always True, and changes to False in specific cases
    
    Returns
    -------
    atlas : nilearn.image.Nifti1Image (atlas file was nib loaded)
    atlas_labels : list of labels for ROIs
    atlas_type : str Choices are: "labels" or "maps". Is further used to choose maskers
    mask_conf : pandas.DataFrame or int.'''

    if atlas_name == "yeo_7":
        atlas_file = datasets.fetch_atlas_yeo_2011()["thick_7"]
        atlas = nib.load(atlas_file)
        atlas_labels = [
            "Visual",
            "Somatosensory",
            "Dorsal Attention",
            "Ventral Attention",
            "Limbic",
            "Frontoparietal",
            "Default",
        ]
        atlas_type = "labels"

    elif atlas_name == "yeo_17":
        atlas_file = datasets.fetch_atlas_yeo_2011()["thick_17"]
        # Missing ROIs correction
        atlas = nib.load(atlas_file)
        atlas_type = "labels"

    elif atlas_name == "difumo64":
        atlas_path = os.path.join(
            main_cwd, "atlases", "atlas_difumo64", "64difumo2mm_maps.nii.gz"
        )
        atlas = nib.load(atlas_path)
        atlas_df = pd.read_csv(
            os.path.join(
                main_cwd, "atlases", "atlas_difumo64", "labels_64_dictionary.csv"
            )
        )
        atlas_labels = atlas_df["Difumo_names"]
        mask_conf = atlas_df.iloc[:, -3:]  # GM WM CSF
        bilat = False
        atlas_type = "maps"

    if bilat == True:
        atlas = make_mask_bilat(atlas)
        if atlas_name == "yeo_7":
            atlas_labels = [
                "L Visual",
                "L Somatosensory",
                "L Dorsal Attention",
                "L Ventral Attention",
                "L Limbic",
                "L Frontoparietal",
                "L Default",
                "R Visual",
                "R Somatosensory",
                "R Dorsal Attention",
                "R Ventral Attention",
                "R Limbic",
                "R Frontoparietal",
                "R Default",
            ]
    if atlas_name == "yeo_17":
        # -- Removing missing ROIs--
        filt_mask = np.array(atlas.dataobj)
        # filt_mask[filt_mask == 9.0] = 0
        # filt_mask[filt_mask == 26.0] = 0  # 9. is the label of this ROI we are removing
        atlas = new_img_like(atlas, filt_mask)
        atlas_labels = np.unique(atlas.get_fdata())  # remove 0

    if atlas_name == "BASC":
        atlas_file = datasets.fetch_atlas_basc_multiscale_2015(
            version="sym", resolution=12
        )
    mask_conf = 0

    return atlas, atlas_labels, atlas_type, mask_conf


def make_mask_bilat(bilateral_mask):
    mask_data = bilateral_mask.get_fdata()
    affine = bilateral_mask.affine

    # Get center X-coord
    x_dim = mask_data.shape[0]
    x_center = int(x_dim / 2)

    # Get left mask
    mask_data_left = mask_data.copy()
    mask_data_left[:x_center, :, :] = 0
    # mask_left = nilearn.image.new_img_like(bilateral_mask, mask_data_left, affine=affine, copy_header=True)

    # Get right mask
    mask_data_right = mask_data.copy()
    mask_data_right[x_center:, :, :] = 0
    # mask_right = nilearn.image.new_img_like(bilateral_mask, mask_data_right, affine=affine, copy_header=True)

    # Labels corrections
    mask_data_right[mask_data_right > 0] += mask_data.max()
    new_bilat_mask_data = mask_data_left + mask_data_right

    return new_img_like(
        bilateral_mask, new_bilat_mask_data, affine=affine, copy_header=True
    )


def compute_cov_measures(correlation_measure, results):
    """
    Computes connectomes based on timeseries from diff. condition : pre, post, contrast.
    As well as mean and individual correl. matrix
    This function applies a R to Z transform and normalizes between 0-1 the weights of the adj matrix
    Returns
    -------
    dict.
        dict. with diff. keys associated with each conditions
    """
    pre_connectomes = correlation_measure.fit_transform(results["pre_series"])
    post_connectomes = correlation_measure.fit_transform(results["post_series"])

    return pre_connectomes, post_connectomes


def proc_connectomes(ls_connectomes,arctanh=False, remove_negw=False, normalize=False):
    """
    arctanh() is the r to Z transformation, then negative weights are removed and the matrix is normalized between 0-1
    """
    proc_ls = []
    for array in ls_connectomes:

        scaler = MinMaxScaler(feature_range=(array.min(), array.max()))

        if arctanh:
            array = np.arctanh(array)  # R to Z transf
        elif remove_negw: # Remove negative edges
            array[array < 0] = 0
        elif normalize:
            shape_save = array.shape
            array.reshape(-1)
            print(array.shape)
            normalized_array = scaler.fit_transform(array)
            array = normalized_array.reshape(shape_save)
        np.fill_diagonal(array, 0)
        proc_ls.append(array)

    return proc_ls


def weight_substraction_postpre(ls_res_post, ls_res_pre):
    """
    Element wise substraction of post - pre connectomes. Applied to weights of the adj. matrix
    *Order of the list of arrays is important*
    return : list of arrays
    """
    res_ls = []
    for post, pre in zip(ls_res_post, ls_res_pre):
        sub_res = post - pre
        res_ls.append(sub_res)

    return res_ls


def extract_features(results):
    """Vectorize each connectivity matrix and saves as a new key in dict. 'results'

    Parameters
    ----------
    result : Dict.
        Dict containing list of individual connectomes from diff. condition (pre, post, contrast)
        Stored as different keys

    Returns
    -------
    Results dict.

    """
    tril_mask = np.tril(np.ones(results["pre_connectomes"].shape[-2:]), k=-1).astype(
        bool
    )

    results["preX"] = np.stack(
        [
            results["pre_connectomes"][i][..., tril_mask]
            for i in range(0, len(results["pre_connectomes"]))
        ],
        axis=0,
    )
    results["postX"] 
    results["contrastX"] = np.stack(
        [
            results["contrast_connectomes"][i][..., tril_mask]
            for i in range(0, len(results["contrast_connectomes"]))
        ],
        axis=0,
    )
    return results


def sym_matrix_to_vec(symmetric, discard_diagonal=True):
    """Return the flattened lower triangular part of an array.

    If diagonal is kept, diagonal elements are divided by sqrt(2) to conserve
    the norm. Acts on the last two dimensions of the array if not 2-dimensional.

    Parameters
    ----------
    symmetric : numpy.ndarray or list of numpy arrays, shape\
        (..., n_features, n_features)
        Input array.

    discard_diagonal : boolean, optional
        If True, the values of the diagonal are not returned.
        Default=False.

    Returns
    -------
    output : numpy.ndarray
        The output flattened lower triangular part of symmetric. Shape is
        (..., n_features * (n_features + 1) / 2) if discard_diagonal is False
        and (..., (n_features - 1) * n_features / 2) otherwise.

    """
    if discard_diagonal:
        # No scaling, we directly return the values
        tril_mask = np.tril(np.ones(symmetric.shape[-2:]), k=-1).astype(bool)
        return symmetric[..., tril_mask]
    scaling = np.ones(symmetric.shape[-2:])
    np.fill_diagonal(scaling, sqrt(2.0))
    tril_mask = np.tril(np.ones(symmetric.shape[-2:])).astype(bool)
    return symmetric[..., tril_mask] / scaling[tril_mask]

    # def reg_model(connectomes)

    features = []


def npsave_features(save_to, results):
    np.save(os.path.join(save_to, f"features_pre"), results["preX"], allow_pickle=True)
    np.save(
        os.path.join(save_to, f"features_post"), results["postX"], allow_pickle=True
    )
    np.save(
        os.path.join(save_to, f"features_contrast"),
        results["contrastX"],
        allow_pickle=True,
    )


def diff_rz(pre_connectome, post_connectome, verbose=True):
    """

    pre (list) : list with all connectomes from pre condition
    post(list) : List with all connectomes from post condition

    """

    diff = list()
    for pre, post in zip(pre_connectome, post_connectome):
        res = np.arctanh(pre) - np.arctanh(post)
        diff.append(res)

    if verbose:
        print(
            "Computing diff in lists of {}, {} connectome with r to Z arctanh func./n Diff matrix has shape : {} ".format(
                len(pre_connectome), len(post_connectome), diff[0].shape
            )
        )

    return diff


# Function definition
def read_data(p, key):
    results = dict()
    results["pre_mean"] = np.load(os.path.join(p, key, "pre_hyp_mean_connectome.npy"))
    results["post_mean"] = np.load(os.path.join(p, key, "post_hyp_mean_connectome.npy"))
    results["contrast_mean"] = np.load(
        os.path.join(p, key, "contrast_mean_connectome.npy")
    )

    return dict(results)


def out(
    root,
    folder,
    atlas_labels,
    atlas,
    atlas_name,
    conditions=None,
    cov_estim="Correlation",
    mat_tresh=None,
    mask_bilat=True,
    con_tresh=None,
    plot_con=True,
):
    results = read_data(root, folder)
    title = "{} at >{}% treshold".format(atlas_name, con_tresh)
    i = 0
    for correlation_matrix, cond in zip(
        [results[condition] for condition in results.keys()], results.keys()
    ):
        if conditions is not None:
            cond = conditions[i]
        mat_title = f"{cond} {cov_estim}"
        np.fill_diagonal(correlation_matrix, 0)
        if mat_tresh is not None:
            correlation_matrix = np.where(
                correlation_matrix < abs(0.3), 0, correlation_matrix
            )

        plotting.plot_matrix(
            correlation_matrix,
            labels=atlas_labels,
            colorbar=True,
            title=mat_title,
            vmax=0.8,
            vmin=-0.8,
        )

        if len(atlas.shape) > 3 and atlas.shape[3] > 1:
            print("plotting probabilistic connectome")
            plotting.plot_connectome(
                correlation_matrix,
                find_probabilistic_atlas_cut_coords(atlas),
                edge_threshold=con_tresh,
                title=f"{cond} {cov_estim} at {con_tresh}",
                display_mode="lzry",
                colorbar=True,
            )
            plotting.show()
        else:
            if plot_con:
                plot_bilat_nodes(
                    correlation_matrix,
                    atlas,
                    title=f"{cond} Correlation at {con_tresh}",
                    tresh=con_tresh,
                    mask_bilat=mask_bilat,
                )
                plotting.show()
        i += 1
    """ 
    if len(atlas.shape) == 3:
        plotting.plot_prob_atlas(atlas, title=title)
    else:
        plot_bilat_nodes(correlation_matrix, atlas, title=title, mask_bilat=True)
        plotting.plot_roi(atlas, title=title)
    """
