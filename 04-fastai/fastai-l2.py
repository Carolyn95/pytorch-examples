# create image dataset through Google images
from fastai.vision import *
"""
# get urls of each of the images (scroll down the page for more images)
# disable ad blocking extensions first, otherwise window.open() command doesn't work
urls = Array.from(document.querySelectorAll('.rg_i')).map(el=> el.hasAttribute('data-src')?el.getAttribute('data-src'):el.getAttribute('data-iurl'));
window.open('data:text/csv;charset=utf-8' + escape(urls.join('\n')));
"""

# create directory and upload files to your server
folder = 'black'
file = 'urls_black.csv'
path = Path('data/bears')
dest = path / folder
dest.mkdir(parents=True, exist_ok=True)

folder = 'teddys'
file = 'urls_teddys.csv'
path = Path('data/bears')
dest = path / folder
dest.mkdir(parents=True, exist_ok=True)

folder = 'grizzly'
file = 'urls_grizzly.csv'
path = Path('data/bears')
dest = path / folder
dest.mkdir(parents=True, exist_ok=True)
path.ls()

# download images, execute below once for one category, difference btw scripy and ipynb
classes = ['teddys', 'grizzly', 'black']
download_images(path / file, dest, max_pics=200)
# if error happened
# download_images(path/file, dest, max_pics=20, max_workers=0)

# remove images that can't be opened
for c in classes:
  print(c)
  verify_images(path / c, delete=True, max_size=500)

# view data
np.random.seed(42)
data = ImageDataBunch.from_folder(path,
                                  train='.',
                                  valid_pct=0.2,
                                  df_tfms=get_transforms(),
                                  size=224,
                                  num_workers=4).normalize(imagenet_stats)
"""
# if data has been cleaned already, use this chunk
np.random.seed(42)
data = ImageDataBunch.from_folder(path, train='.', valid_pct=0.2, ds_tfms=get_transforms(), size=224, num_workers=4).normalize(imagenet_stats)
"""

print(data.classes)
data.show_batch(rows=3, figsize=(7, 8))
print('{}, {}, {}, {}'.format(data.classes, data.c, len(data.train_ds),
                              len(data.valid_ds)))

# train model
learn = cnn_learner(data, models.resnet34, metrics=error_rate)
learn.fit_one_cycle(4)
learn.save('stage-1')
learn.unfreeze()
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(2, max_lr=slice(3e-5, 3e-4))
learn.save('stage-2')

# interpretation
learn.load('stage-2')
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()

# don't remove original, create new processed instead
db = (ImageList.from_folder(path).split_none().label_from_folder().transform(
    get_transforms(), size=224).databunch())

# create new learner with all the images
learn_cln = cnn_learner(db, models.resnet34, metrics=error_rate)
learn_cln.load('stage-2')
ds, idxs = DataFormatter().from_toplosses(learn_cln)
ImageCleaner(df, idxs, path)
ds, idxs = DatasetFormatter().from_similars(learn_cln)
ImageCleaner(df, idxs, path, duplicates=True)

# put the model in production
learn.export()
# user cpu for inference
defaults.device = torch.device('cpu')
img = open_image(path / 'black' / '00000021.jpg')
img

learn = load_learner(path)
pred_class, pred_idx, outputs = learn.predict(img)
pred_class.obj
"""
# example of using a route, typically used in a web app toolkit
@app.route("/classify-url", methods=["GET"])
async def classify_url(request):
  bytes = await get_bytes(request.query_params["url"])
  img = open_image(BytesIO(bytes))
  _, _, losses = learner.predict(img)
  return JSONResponse({
      "predictions":
          sorted(zip(cat_learner.data.classes, map(float, losses)),
                 key=lambda p: p[1],
                 reverse=True)
  })
"""

# things may need to tune, coz they can easily go wrong
# learning rate & number of epochs
# learning rate too high, learning rate too low, plot losses
# learning rate too low not only train slow but also causes overfit because it is getting too many looks at each image
learn = cnn_learner(data, models.resnet34, metrics=error_rate)
learn.fit_one_cycle(1, max_lr=0.5)
learn = cnn_learner(data, models.resnet34, metrics=error_rate)
learn.fit_one_cycle(5, max_lr=1e-5)
learn.recorder.plot_losses()

# too few epochs & too many epochs
learn = cnn_learner(data, models.resnet34, metrics=error_rate, pretrained=False)
learn.fit_one_cycle()
np.random.seed(42)
data = ImageDataBunch.from_folder(path,
                                  train='.',
                                  valid_pct=0.9,
                                  bs=32,
                                  ds_tfms=get_transforms(do_flip=False,
                                                         max_rotate=0,
                                                         max_zoom=1,
                                                         max_lighting=0,
                                                         max_warp=0),
                                  size=224,
                                  num_workers=4).normalize(imagenet_stats)
learn = cnn_learner(data, models.resnet50, metrics=error_rate, ps=0, wd=0)
learn.unfreeze()
learn.fit_one_cycle(40, slice(1e-6, 1e-4))

# SGD - Stochastic Gradient Descent, optimization method
from fastai.basics import *
n = 100
x = torch.ones(n, 2)
x[:, 0].uniform_(-1., 1)
print(x[:5])
a = tensor(3., 2)
print(a)
y = x @ a + 0.25 * torch.randn(n)
plt.scatter(x[:, 0], y)


# target is to find a that minimize the error between the points(x,y) and the line x@a
# this a is different from above a
# most common loss function / error function is mse (mean squared error)
def mse(y_hat, y):
  return ((y_hat - y)**2).mean()


# to calculate prediction, aka, y_hat, assume a = (-1.0, 1.0)
a = tensor(-1.0, 1)
y_hat = x @ a
print(mse(y_hat, y))

plt.scatter(x[:, 0], y)
plt.scatter(x[:, 0], y_hat)

# how to find the best values of a => how to find the best fitting regression
# minimize mse_loss
# gradient descent is an algorithm that minimizes functions, given a function defined by a set of parameters, gradient descent starts with an initial set
# of parameter values and iteractively moves towards a set of parameter values that minimize the function.
# This iteractive minimization is achieved by taking steps in the negative direction of the function gradient.
a = nn.Parameter(a)
print(a)


def update():
  y_hat = x @ a
  loss = mse(y_hat, y)
  if t % 10 == 0:
    print(loss)
  loss.backward()
  with torch.no_grad():
    a.sub_(lr * a.grad)
    a.grad.zero_()


lr = 1e-1
for t in range(100):
  update()

plt.scatter(x[:, 0], y)
plt.scatter(x[:, 0], x @ a.detach())

# bonus, annimate it
from matplotlib import animation, rc
rc('animation', html='jshtml')
a = nn.Parameter(tensor(-1., 1))
fig = plt.figure()
plt.scatter(x[:, 0], y, c='orange')
line, = plt.plot(x[:, 0], x @ a.detach())
plt.close()


def animate(i):
  update()
  line.set_ydata(x @ a.detach())
  return line


animation.FuncAnimation(fig, animate, np.arange(0, 100), interval=20)
