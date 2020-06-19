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

# plot confusion matrix to see which category the model is most confused with the other
interp = ClassificationInterpretation.from_learner(learn)
losses, idxs = interp.top_lossess()
len(data.valid_ds) == len(losses) == (idxs)
interp.plot_top_losses(9, figsize=(15, 11))
interp.plot_confusion_matrix(figsize=(12, 12), dpi=60)
interp.most_confused(min_val=2)

# check a few first, have a general idea in mind, decide to move on
learn.unfreeze()
learn.fit_one_cycle(1)
learn.load('stage-1')
learn.lr_find()
learn.recorder.plot()
learn.unfreeze()
learn.fit_one_cycle(2, max_lr=slice(1e-6, 1e-4))

# train with resnet50
# larger network requires more GPU memory, larger size of image requires more GPU memory
data = ImageDataBunch.from_name_re(path_img,
                                   fnames,
                                   pat,
                                   ds_tfms=get_transforms(),
                                   size=299,
                                   bs=bs // 2).normalize(imagenet_stats)

learn = cnn_learner(data, models.resnet50, metrics=error_rate)
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(8)
learn.save('stage-1-50')
learn.unfreeze()
learn.fit_one_cycle(3, max_lr=slice(1e-6, 1e-4))
learn.load('stage-1-50')
interp = ClassificationInterpretation.from_learner(learn)
interp.most_confused(min_val=2)

# mnist data
path = untar_data(URLs.MNIST_SAMPLE)
print(path)
tfms = get_transforms(do_flip=False)
data = ImageDataBunch.from_folder(path, df_tfms=tfms, size=26)
data.show_batch(rows=3, figsize=(5, 5))
learn = cnn_learner(data, models.resnet18, metrics=accuracy)
learn.fit(2)
df = pd.read_csv(path / 'label.csv')
print(df.head())
data = ImageDataBunch.from_csv(path, ds_tfms=tfms, size=28)
data.show_batch(rows=3, figsize=(5, 5))
print(data.classes)

data = ImageDataBunch.from_df(path, df, ds_tfms=tfms, size=24)
print(data.classes)

fn_paths = [path / name for name in df['name']]
fn_paths[:2]

pat = r"/(\d)/\d+\.png$"
data = ImageDataBunch.from_name_re(path,
                                   fn_paths,
                                   pat=pat,
                                   ds_tfms=tfms,
                                   size=24)
print(data.classes)

data = ImageDataBunch.from_name_func(path,
                                     fn_paths,
                                     ds_tfms=tfms,
                                     size=24,
                                     label_func=lambda x: '3'
                                     if '/3/' in str(x) else '7')
print(data.classes)

labels = [('3' if '/3/' in str(x) else '7') for x in fn_paths]
print(labels[:5])

data = ImageDataBunch.from_lists(path,
                                 fn_paths,
                                 labels=labels,
                                 ds_tfms=tfms,
                                 size=24)
print(data.classes)