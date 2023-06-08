import os
import glob as glob
import pandas as pd
import numpy as np
import nibabel as nib
from sklearn.utils import Bunch
from nilearn import plotting
from nilearn.image import new_img_like, load_img
from nilearn import datasets


def load_data(path):
    """
    Load subject information into memory

    """
    data = Bunch(
        subjects=[sub for sub in os.listdir(path) if "APM" in sub],
        pre_hyp=[
            glob.glob(os.path.join(path, sub, "*before*", "*4D*"))[0]
            for sub in os.listdir(path)
            if "APM" in sub
        ],
        post_hyp=[
            glob.glob(os.path.join(path, sub, "*during*", "*4D*"))[0]
            for sub in os.listdir(path)
            if "APM" in sub
        ],
        anat=[
            glob.glob(os.path.join(path, sub, "anatomy", "*.nii"))[0]
            for sub in os.listdir(path)
            if "APM" in sub
        ],
        phenotype=pd.DataFrame(
            pd.read_excel(
                glob.glob(os.path.join(path, "*variables*"))[0],
                sheet_name=0,
                index_col=1,
                header=2,
            )
        ),
    )

    return data


def load_choose_atlas(atlas_name, bilat=True):
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

    elif atlas_name == "yeo_17":
        atlas_file = datasets.fetch_atlas_yeo_2011()["thick_17"]
        # Missing ROIs correction
        atlas = nib.load(atlas_file)
    elif atlas_name == "difumo":
        atlas_path = r"C:\Users\Dylan\Desktop\UM_Bsc_neurocog\E22\Projet_Ivado_rainvillelab\connectivity_project\resting_state_hypnosis\atlases\atlas_difumo64\64difumo2mm_maps.nii.gz"
        atlas = nib.load(atlas_path)
        atlas_df = pd.read_csv(
            r"C:\Users\Dylan\Desktop\UM_Bsc_neurocog\E22\Projet_Ivado_rainvillelab\connectivity_project\resting_state_hypnosis\atlases\atlas_difumo64\labels_64_dictionary.csv"
        )
        atlas_labels = atlas_df["Difumo_names"]
        confounds = atlas_df.iloc[:, -3:]  # GM WM CSF
        bilat = False

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
        filt_mask[filt_mask == 9.0] = 0
        filt_mask[filt_mask == 26.0] = 0  # 9. is the label of this ROI we are removing
        atlas = new_img_like(atlas, filt_mask)
        atlas_labels = np.unique(atlas.get_fdata())[1:]  # remove 0

    if atlas_name == "BASC":
        atlas_file = datasets.fetch_atlas_basc_multiscale_2015(
            version="sym", resolution=12
        )
    if atlas_name != "difumo":
        confounds = None
    return atlas, atlas_labels, confounds


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


def plot_bilat_nodes(
    correlation_matrix, atlas_file, title, mask_bilat=False, reduce_roi=False
):
    atlas = nib.load(atlas_file)
    if reduce_roi:
        load_mask = nib.load(atlas_file)
        filt_mask = np.array(load_mask.dataobj)
        filt_mask[filt_mask == 9.0] = 0  # 9. is the label of this ROI we are removing
        atlas_file = filt_mask
    if mask_bilat:
        atlas = make_mask_bilat(nib.load(atlas_file))

    left_coordinates = plotting.find_parcellation_cut_coords(labels_img=atlas)
    right_coordinates = plotting.find_parcellation_cut_coords(
        labels_img=atlas, label_hemisphere="right"
    )
    print(left_coordinates.shape)

    plot_connectome(
        correlation_matrix,
        left_coordinates,
        right_coordinates,
        edge_threshold=None,
        title=title,
    )
    plotting.show()


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


from matplotlib import cm
from nilearn.plotting import plot_glass_brain


def plot_connectome(
    adjacency_matrix,
    node_coords_left,
    node_coords_right,
    node_color="auto",
    node_size=50,
    edge_cmap=cm.bwr,
    edge_vmin=None,
    edge_vmax=None,
    edge_threshold=None,
    output_file=None,
    display_mode="ortho",
    figure=None,
    axes=None,
    title=None,
    annotate=True,
    black_bg=False,
    alpha=0.7,
    edge_kwargs=None,
    node_kwargs=None,
    colorbar=False,
):
    """Plot connectome on top of the brain glass schematics.

    The plotted image should be in MNI space for this function to work
    properly.

    In the case of 'l' and 'r' directions (for hemispheric projections),
    markers in the coordinate x == 0 are included in both hemispheres.

    Parameters
    ----------
    adjacency_matrix : numpy array of shape (n, n)
        Represents the link strengths of the graph. The matrix can be
        symmetric which will result in an undirected graph, or not
        symmetric which will result in a directed graph.

    node_coords : numpy array_like of shape (n, 3)
        3d coordinates of the graph nodes in world space.

    node_color : color or sequence of colors or 'auto', optional
        Color(s) of the nodes. If string is given, all nodes
        are plotted with same color given in string.

    node_size : scalar or array_like, optional
        Size(s) of the nodes in points^2. Default=50.

    edge_cmap : colormap, optional
        Colormap used for representing the strength of the edges.
        Default=cm.bwr.

    edge_vmin, edge_vmax : float, optional
        If not None, either or both of these values will be used to
        as the minimum and maximum values to color edges. If None are
        supplied the maximum absolute value within the given threshold
        will be used as minimum (multiplied by -1) and maximum
        coloring levels.

    edge_threshold : str or number, optional
        If it is a number only the edges with a value greater than
        edge_threshold will be shown.
        If it is a string it must finish with a percent sign,
        e.g. "25.3%%", and only the edges with a abs(value) above
        the given percentile will be shown.
    %(output_file)s
    display_mode : string, optional
        Choose the direction of the cuts: 'x' - sagittal, 'y' - coronal,
        'z' - axial, 'l' - sagittal left hemisphere only,
        'r' - sagittal right hemisphere only, 'ortho' - three cuts are
        performed in orthogonal directions. Possible values are: 'ortho',
        'x', 'y', 'z', 'xz', 'yx', 'yz', 'l', 'r', 'lr', 'lzr', 'lyr',
        'lzry', 'lyrz'. Default='ortho'.
    %(figure)s
    %(axes)s
    %(title)s
    %(annotate)s
    %(black_bg)s
        Default=False.
    alpha : float between 0 and 1, optional
        Alpha transparency for the brain schematics. Default=0.7.

    edge_kwargs : dict, optional
        Will be passed as kwargs for each edge matlotlib Line2D.

    node_kwargs : dict, optional
        Will be passed as kwargs to the plt.scatter call that plots all
        the nodes in one go.
    %(colorbar)s
        Default=False.

    See Also
    --------
    nilearn.plotting.find_parcellation_cut_coords : Extraction of node
        coords on brain parcellations.
    nilearn.plotting.find_probabilistic_atlas_cut_coords : Extraction of
        node coords on brain probabilistic atlases.

    """
    display = plot_glass_brain(
        None,
        display_mode=display_mode,
        figure=figure,
        axes=axes,
        title=title,
        annotate=annotate,
        black_bg=black_bg,
        alpha=alpha,
    )

    display.add_graph(
        adjacency_matrix,
        node_coords_left,
        node_color=node_color,
        node_size=node_size,
        edge_cmap=edge_cmap,
        edge_vmin=edge_vmin,
        edge_vmax=edge_vmax,
        edge_threshold=edge_threshold,
        edge_kwargs=edge_kwargs,
        node_kwargs=node_kwargs,
        colorbar=colorbar,
    )

    display.add_graph(
        adjacency_matrix,
        node_coords_right,
        node_color=node_color,
        node_size=node_size,
        edge_cmap=edge_cmap,
        edge_vmin=edge_vmin,
        edge_vmax=edge_vmax,
        edge_threshold=edge_threshold,
        edge_kwargs=edge_kwargs,
        node_kwargs=node_kwargs,
        colorbar=colorbar,
    )

    if output_file is not None:
        display.savefig(output_file)
        display.close()
        display = None

    return display


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
