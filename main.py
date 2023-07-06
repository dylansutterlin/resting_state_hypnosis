from main_con import con_matrix

p = r"C:\Users\Dylan\Desktop\UM_Bsc_neurocog\E22\Projet_Ivado_rainvillelab\connectivity_project\resting_state_hypnosis\CBF_normalized"
else_dir = (
    r"E:\Users\Dylan\Desktop\UdeM_H22\E_PSY3008\data_desmartaux\wHYPNOSIS_ASL_ANALYSIS"
)
save_base = r"C:\Users\Dylan\Desktop\UM_Bsc_neurocog\E22\Projet_Ivado_rainvillelab\results\results_con"
# atlas_name = ["yeo_7", "difumo64"]

con_matrix(
    p,
    else_dir=else_dir,
    save_base=save_base,
    save_folder="difumo64_correlation",
    atlas_name="difumo64",
    sphere_coord=False,
    connectivity_measure="correlation",
    verbose=False,
)
