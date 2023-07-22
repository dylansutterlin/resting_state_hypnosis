import nibabel as nib
import numpy as np
from nilearn import plotting
from nilearn.image import mean_img
import os
import glob as glob

path = r"C:\Users\Dylan\Desktop\UM_Bsc_neurocog\E22\Projet_Ivado_rainvillelab\connectivity_project\resting_state_hypnosis\CBF_normalized"

dirs = glob.glob(os.path.join(path, "APM*", "*"))
for i, f in enumerate(dirs):
    filename = f[-39:]
    img = mean_img(nib.load(f))
    plotting.plot_roi(img, title=filename)
    plotting.show()

    from nilearn.maskers import NiftiMasker

    m = NiftiMasker(mask_strategy="whole-brain-template")
    m.fit(img)
    report = m.generate_report()
    os.chdir(
        r"C:\Users\Dylan\Desktop\UM_Bsc_neurocog\E22\Projet_Ivado_rainvillelab\connectivity_project\resting_state_hypnosis\report_test"
    )
    report.save_as_html("report_{}.html".format(i))
