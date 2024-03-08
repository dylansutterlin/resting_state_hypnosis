import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import stats
import numpy as np
import os

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


def plot_matrices(cov, prec, tresh, labels):
    """Plot covariance and precision matrices, for a given processing."""
    prec = prec.copy()  # avoid side effects

    # Put zeros on the diagonal, for graph clarity.
    size = prec.shape[0]
    prec[list(range(size)), list(range(size))] = 0
    span = max(abs(prec.min()), abs(prec.max()))

    # Display covariance matrix
    plotting.plot_matrix(
        cov,
        cmap=plotting.cm.bwr,
        vmin=-1,
        vmax=1,
        title=f"covariance at {tresh} treshold (only highest values)",
        labels=labels,
    )
    # Display precision matrix
    plotting.plot_matrix(
        prec,
        cmap=plotting.cm.bwr,
        vmin=-span,
        vmax=span,
        title=f"precision at {tresh} treshold (only highest values)",
        labels=labels,
    )


def plot_bilat_nodes(
    correlation_matrix, atlas, title, tresh=None, mask_bilat=False, reduce_roi=False
):
    if reduce_roi:
        filt_mask = np.array(atlas.dataobj)
        filt_mask[filt_mask == 9.0] = 0  # 9. is the label of this ROI we are removing
        atlas_file = filt_mask
    if mask_bilat:
        atlas = make_mask_bilat(atlas)

    left_coordinates = plotting.find_parcellation_cut_coords(labels_img=atlas)
    right_coordinates = plotting.find_parcellation_cut_coords(
        labels_img=atlas, label_hemisphere="right"
    )

    plot_connectome(
        correlation_matrix,
        left_coordinates,
        right_coordinates,
        edge_threshold=tresh,
        title=title,
        display_mode="lzry",
        colorbar=True,
    )
    plotting.show()

    def graphLasso_covariance_estim(series, cond, atlas_name="yeo_7", tresh=None):
        atlas, labels, _ = load_choose_atlas(atlas_name, bilat=True)
        gl = GraphicalLassoCV(verbose=1)
        gl.fit(np.concatenate(series))
        atlas_region_coords = plotting.find_parcellation_cut_coords(labels_img=atlas)

        # ---------------------------------------------------------------------

        # Display the covariance
        plotting.plot_connectome(
            gl.covariance_,
            atlas_region_coords,
            edge_threshold=tresh,
            title=f"{cond} Covariance  at {tresh}",
            display_mode="lzry",
            colorbar=True,
        )
        plotting.plot_connectome(
            -gl.precision_,
            atlas_region_coords,
            edge_threshold=tresh,
            title=f"{cond} : Sparse inverse covariance at {tresh} ",
            display_mode="lzry",
            edge_vmax=0.5,
            edge_vmin=-0.5,
            colorbar=True,
        )
        plot_matrices(gl.covariance_, gl.precision_, tresh, labels)

        plotting.show()


def dist_mean_edges(cond, matrix_list, save_to ):
    ''' Function to save distribution plots of mean connectomes edges weights

    Parameters
    ----------
    cond : str
        Condition name used to save file name
    matrix_list : list
        List of connectomes
    save_to : str
        Path to save the plots

    '''
    
    print('---Saving edges weights distribution')
    adj_matrix = np.mean(np.stack(matrix_list, axis=-1), axis=-1)
    np.fill_diagonal(adj_matrix, np.nan)
    fig = sns.heatmap(adj_matrix, cmap="coolwarm", square=False)
    fig.get_figure().savefig(os.path.join(save_to, f'fig_heatMapCM-{cond}.png'))
    plt.close()

    # Weight distribution plot
    bins = np.arange(np.sqrt(len(np.concatenate(adj_matrix))))
    bins = (bins-np.min(bins))/np.ptp(bins)
    fig, axes = plt.subplots(1,2, figsize=(15,5))
    rawdist = sns.histplot(adj_matrix.flatten(), bins=bins, kde=False, ax=axes[0])
    rawdist.set(xlabel='Edge correlations', ylabel='Density Frequency', title='Raw edge weights distribution')

    log10dist = sns.histplot(np.log10(adj_matrix.flatten()), kde=False, ax=axes[1], stat='density')
    log10dist.set(xlabel='Log10 edge correlations', title='Log10 edge weights distribution')
    plt.savefig(os.path.join(save_to, f'fig_weightDist-{cond}.png'))
    plt.close()

    #plt.plot(results_con['pre_series'][0][43], label=labels[43])
        #plt.title("POTime Series")
        #plt.xlabel("Scan number")
        #plt.ylabel("non-Normalized signal")
        #plt.legend()
        #plt.tight_layout()
    
def visu_correl(vd, vi, save_to, vd_name, vi_name, title):
    corr_coeff, p_value = stats.pearsonr(vd, vi)
    r_squared = np.corrcoef(np.array(vd), np.array(vi))[0, 1]**2
    # Scatter plot of vd score vs mean rCBF diff
    plt.scatter(vd, vi)
    regression_params = np.polyfit(vd, vi, 1)
    regression_line = np.polyval(regression_params, vd)

    # Plot the regression line
    plt.plot(vd, regression_line, color='red', linewidth=2, label='Regression Line')

    plt.xlabel(f'{vd_name} score')
    plt.ylabel(f'{vi_name}')
    plt.title(title)
    text = f'Correlation: {corr_coeff:.2f}\nP-value: {p_value:.4f}\nR-squared: {r_squared:.2f}'
    plt.annotate(text, xy=(0.05, 0.85), xycoords='axes fraction', fontsize=10, ha='left', va='top')
    plt.savefig(os.path.join(save_to, 'fig_autoXrCBF-correl.png'))
    plt.close()
