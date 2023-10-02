from scripts import main_con as main

p = r"/home/p1226014/projects/def-rainvilp/p1226014/hypnosis_rest/CBF_normalized/"
conf_dir = (
    r"/home/p1226014/projects/def-rainvilp/p1226014/data/rainville2019_ASL_preproc"
)
save_base = r"/home/p1226014/projects/def-rainvilp/p1226014/data/rainville2019_ASL_preproc/results"

# p = r"/data/rainville//HYPNOSIS_ASL_ANALYSIS/CBF_normalized/"
# else_dir = r"/data/rainville/HYPNOSIS_ASL_ANALYSIS"
# save_base = r"/data/rainville/dylanSutterlin/results/connectivity"
# atlas_name = ["yeo_7", "difumo64"]


# )
# save_base = r"C:\Users\Dylan\Desktop\UM_Bsc_neurocog\E22\Projet_Ivado_rainvillelab\results\results_con"
# atlas_name = ["yeo_7", "difumo64"]

main.con_matrix(
    p,
    conf_dir=conf_dir,
    save_base=save_base,
    cwd=os.getcwd(),
    save_folder="difumo64_tangentZ",
    atlas_name="difumo64",
    sphere_coord=[
        (54, -28, 26),
        (-20, -26, -14),
        (-2, 20, 32),
        (-8, 44, 28),
        (-6, -26, 46),
    ],
    connectivity_measure="tangent",
    verbose=False,
)
