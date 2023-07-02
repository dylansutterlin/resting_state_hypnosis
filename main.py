from main_con import con_matrix

p = r"E:\Users\Dylan\Desktop\UdeM_H22\E_PSY3008\data_desmartaux\HYPNOSIS_ASL_ANALYSIS"
# p = r"E:\Users\Dylan\Desktop\UdeM_H22\E_PSY3008\data_desmartaux\2sub_ASL_data"
save_base = r"C:\Users\Dylan\Desktop\UM_Bsc_neurocog\E22\Projet_Ivado_rainvillelab\results\results_con"
# atlas_name = ["yeo_7", "difumo64"]

con_matrix(
    p,
    save_base=save_base,
    save_folder="difumo64_precision",
    atlas_name="difumo64",
    sphere_coord=False,
    connectivity_measure="precision",
    verbose=False,
)
