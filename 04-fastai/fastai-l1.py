from fastai.vision import *
from fastai.metrics import error_rate

bs = 64

path = untar_data(URLs.PETS)
print(path)
