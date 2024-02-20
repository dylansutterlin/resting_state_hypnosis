from PIL import Image
import os
import glob as glob
dir = os.path.join(os.getcwd(),'/debug/difumo64_correlation')
imgs_name = glob.glob(os.path.join(dir, '*.png'))
from simshow import simshow

for imgi in imgs_name:
    with Image.open(imgi) as img:
        img.show()
    simshow(imgi)

