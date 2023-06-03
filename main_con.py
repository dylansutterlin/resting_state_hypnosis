import argparse

import os
import glob
import numpy as np
import nibabel as nib
from nilearn.maskers import NiftiMapsMasker, NiftiLabelsMasker, NiftiMasker
from nilearn.connectome import ConnectivityMeasure
import func
from nilearn import datasets, plotting, image
from nilearn.regions import connected_label_regions
from nilearn.signal import clean as signal_clean


def con_matrix(
    data_dir,
    save_base,
    save_folder,
    atlas_name="yeo_7",
    atlas_type="labels",
    sphere_coord=None,
    connectivity_measure="correlation",
    plot_atlas=False,
    verbose=False,
):
    """_summary_

    Parameters
    ----------
    data_dir : str
        Path to data
    save_base : str
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
    conditions = ["pre_hyp", "post_hyp", "contrast"]
    pre_data = data.pre_hyp
    post_data = data.post_hyp
    results = dict(pre_series=list(), post_series=list())

    # --Atlas choices--
    atlas, atlas_labels, confounds = func.load_choose_atlas(atlas_name, bilat=True)

    # --Labels--
    # region_labels = connected_label_regions(atlas)

    # --Masker parameters--
    if atlas_type == "maps":
        masker = NiftiMapsMasker(
            maps_img=atlas,
            mask_imgs=atlas_labels,
            t_r=3,
            high_pass=0.1,
            standardize="zscore_sample",
            memory="nilearn_cache",
            verbose=0,
        )
    elif atlas_type == "labels":
        # labels = atlas.labels
        masker = NiftiLabelsMasker(
            labels_img=atlas,
            labels=atlas_labels,
            standardize="zscore_sample",
            resampling_target="data",
        )
        print(" Labeled masker!")
    elif atlas_type == "sphere_seed":
        sphere_masker = NiftiSpheresMasker(
            sphere_coord, radius=8, mask_img=None, high_pass=0.1
        )

    voxel_masker = NiftiMasker(high_pass=0.1, t_r=3, standardize=True, smoothing_fwhm=6)

    # --Timeseries : Fit and apply mask--
    results["pre_series"] = [masker.fit_transform(ts) for ts in pre_data]
    results["post_series"] = [masker.fit_transform(ts) for ts in post_data]
    print([ts.shape for ts in results["pre_series"]])
    print([ts.shape for ts in results["post_series"]])
    gb_signal = signal_clean(
        np.array(results["pre_series"])
        .mean(axis=1)
        .reshape([np.array(results["pre_series"]).shape[0], 1]),
        high_pass=0.1,
        t_r=3,
        standardize="zscore_sample",
    )

    results["pre_series"] = voxel_masker.fit_transform(pre_data, confounds=gb_signal)
    gb_signal = signal_clean(
        results["post_series"]
        .mean(axis=1)
        .reshape([results["post_series"].shape[0], 1]),
        high_pass=0.1,
        t_r=3,
        standardize="zscore_sample",
    )
    results["post_series"] = voxel_masker.fit_transform(post_data, confounds=gb_signal)
    print([ts.shape for ts in results["pre_series"]])
    print([ts.shape for ts in results["post_series"]])

    # Filter TS !!!
    # for i, shapes, sub in enumerate()
    # idcs = [idcs for idcs, index in enumerate(data.phenotype.index) if index in rename_sub] # Check indices of Y[i] of sub included in analysis
    # y_auto = np.array(y_full_auto[idcs])

    # -- Covariance Estimation--
    correlation_measure = ConnectivityMeasure(
        kind=connectivity_measure, discard_diagonal=True
    )

    results["pre_connectomes"] = correlation_measure.fit_transform(
        results["pre_series"]
    )
    results["pre_mean_connectome"] = correlation_measure.mean_
    results["post_connectomes"] = correlation_measure.fit_transform(
        results["post_series"]
    )
    results["post_mean_connectome"] = correlation_measure.mean_
    results["zcontrast_mean_connectome"] = np.arctanh(
        results["post_mean_connectome"]
    ) - np.arctanh(results["pre_mean_connectome"])
    results["contrast_connectomes"] = [
        post - pre
        for post, pre in zip(results["post_connectomes"], results["pre_connectomes"])
    ]

    # --Save--
    if os.path.exists(os.path.join(save_base, save_folder)) is False:
        os.mkdir(os.path.join(save_base, save_folder))
        save_to = os.path.join(save_base, save_folder)

    for idx, sub in enumerate(data.subjects):
        np.save(
            os.path.join(save_to, f"{sub}_{conditions[0]}_connectomes"),
            results["pre_connectomes"][idx],
            allow_pickle=True,
        )
        np.save(
            os.path.join(save_to, f"{sub}_{conditions[1]}_connectomes"),
            results["post_connectomes"][idx],
            allow_pickle=True,
        )
        np.save(
            os.path.join(save_to, f"{sub}_{conditions[2]}_connectomes"),
            results["contrast_connectomes"][idx],
            allow_pickle=True,
        )
    np.save(
        os.path.join(save_to, f"{conditions[0]}_mean_connectome"),
        results["pre_mean_connectome"],
        allow_pickle=True,
    )
    np.save(
        os.path.join(save_to, f"{conditions[1]}_mean_connectome"),
        results["post_mean_connectome"],
        allow_pickle=True,
    )
    np.save(
        os.path.join(save_to, f"{conditions[2]}_mean_connectome"),
        results["zcontrast_mean_connectome"],
        allow_pickle=True,
    )

    # --Stats--
    matrices = np.asarray(
        [
            np.load(
                os.path.join(save_to, f"{sub}_{conditions[2]}_connectomes.npy"),
                allow_pickle=True,
            )
            for sub in data.subjects
        ]
    )
    y_full_auto = data.phenotype["Unnamed: 68"]  # abs. diff. in perceived automaticity
    # Access selected sub based on id in y
    rename_sub = [f"APM{num}" for num in [sub[4:6] for sub in data.subjects]]
    idcs = [
        idcs for idcs, index in enumerate(data.phenotype.index) if index in rename_sub
    ]
    y_auto = np.array(y_full_auto[idcs])

    # --X/features (vectorize each connectome)--
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
    results["postX"] = np.stack(
        [
            results["post_connectomes"][i][..., tril_mask]
            for i in range(0, len(results["post_connectomes"]))
        ],
        axis=0,
    )
    results["contrastX"] = np.stack(
        [
            results["contrast_connectomes"][i][..., tril_mask]
            for i in range(0, len(results["contrast_connectomes"]))
        ],
        axis=0,
    )

    np.save(os.path.join(save_to, f"features_pre"), results["preX"], allow_pickle=True)
    np.save(
        os.path.join(save_to, f"features_post"), results["postX"], allow_pickle=True
    )
    np.save(
        os.path.join(save_to, f"features_contrast"),
        results["contrastX"],
        allow_pickle=True,
    )
    np.save(os.path.join(save_to, f"Y"), y_auto, allow_pickle=True)

    # --Prints and plot--
    if verbose:
        print(atlas.shape)
        print(np.unique(atlas.get_fdata(), return_counts=True))
        for correlation_matrix in [
            results["pre_mean_connectome"],
            results["post_mean_connectome"],
            results["zcontrast_mean_connectome"],
        ]:  # [results['pre_mean_connetomes'], results['post_mean_connetomes']]:
            np.fill_diagonal(correlation_matrix, 0)
            plotting.plot_matrix(
                correlation_matrix,
                labels=atlas_labels,
                colorbar=True,
                vmax=0.8,
                vmin=-0.8,
            )
            func.plot_bilat_nodes(correlation_matrix, atlas, atlas_name)

        plotting.plot_roi(atlas, title=atlas_name)


"""
# plot connectivity matrix

    matrix = np.load(os.path.join('data/derivatives/connectomes', os.listdir('data/derivatives/connectomes')[0]))
    plotting.plot_matrix(squareform(matrix), vmin = -1, vmax = 1, labels=masker.labels_)
    plt.savefig('results/plots/connectivity_matrix.svg', format='svg')
    plt.clf()
"""

p = r"E:\Users\Dylan\Desktop\UdeM_H22\E_PSY3008\data_desmartaux\HYPNOSIS_ASL_DATA"
save_base = r"C:\Users\Dylan\Desktop\UM_Bsc_neurocog\E22\Projet_Ivado_rainvillelab\results\results_con/"
con_matrix(
    p,
    save_base,
    save_folder="difumo64_correlation",
    atlas_name="difumo64",
    atlas_type="maps",
    connectivity_measure="correlation",
    verbose=True,
)
