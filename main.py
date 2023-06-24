from main_con import con_matrix

p = r"E:\Users\Dylan\Desktop\UdeM_H22\E_PSY3008\data_desmartaux\HYPNOSIS_ASL_DATA"
p = r"C:\Users\Dylan\Desktop\UM_Bsc_neurocog\E22\Projet_Ivado_rainvillelab\test_dataset\test_data_ASL"
save_base = r"C:\Users\Dylan\Desktop\UM_Bsc_neurocog\E22\Projet_Ivado_rainvillelab\results\results_con"
# atlas_name = ["yeo_7", "difumo64"]

con_matrix(
    p,
    save_base=save_base,
    save_folder="seedPO_voxels_correlation",
    atlas_name="difumo64",
    atlas_type="maps",
    sphere_coord=True,
    connectivity_measure="correlation",
    verbose=False,
)
