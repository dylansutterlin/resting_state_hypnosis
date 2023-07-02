import nibabel as nib
import numpy as np
from nilearn import plotting
from nilearn.image import mean_img

file = r"E:\Users\Dylan\Desktop\UdeM_H22\E_PSY3008\data_desmartaux\HYPNOSIS_ASL_ANALYSIS\APM_02_H2\02-PCASL_before_hypnosis\wcbf_0_srASL_4D_during_4D.nii"
img = mean_img(nib.load(file))

plotting.plot_epi(img)
plotting.show()
