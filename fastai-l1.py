from fastai.vision import *
from fastai.metrics import error_rate

bs = 64

path = untar_data(URLs.PETS)
print(path)
print(path.ls())

path_anno = path / 'annotations'
path_img = path / 'images'

# print out paths to data to understand data file structure
fnames = get_image_files(path_img)
print(fnames[:5])
