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

# set random seed to ensure same split for validation set
np.random.seed(2)
pat = r'/([^/]+)_\d+.jpg$'

data = ImageDataBunch.from_name_re(patch_img,
                                   fnames,
                                   pat,
                                   ds_tfms=get_transforms(),
                                   size=224,
                                   bs=bs).normalize(imagenet_stats)
data.show_batch(rows=3, figsize=(7, 6))
print(data.classes)
print(len(data.classes), data.c)  # data.c is equivelant to len(data.classes)

learn = cnn_learner(data, models.resnet34, metrics=error_rate)
print(learn.model)

learn.fit_one_cycle(
    4)  # 4 epochs, print out train_loss, valid_loss and error_rate
learn.save('stage-1')
