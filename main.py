p = r"E:\Users\Dylan\Desktop\UdeM_H22\E_PSY3008\data_desmartaux\HYPNOSIS_ASL_DATA"
save_base = r"C:\Users\Dylan\Desktop\UM_Bsc_neurocog\E22\Projet_Ivado_rainvillelab\results\results_con/"
# atlas_name = ["yeo_7", "difumo64"]

con_matrix(
    p,
    save_base,
    save_folder="yeo7_correlation",
    atlas_name="yeo_7",
    atlas_type="labels",
    connectivity_measure="correlation",
    verbose=True,
)
