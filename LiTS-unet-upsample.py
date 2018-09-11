import os
from fastai.conv_learner import *
from fastai.dataset import *
from fastai.models.resnet import vgg_resnet50, resnet34

torch.cuda.set_device(0)
torch.backends.cudnn.benchmark = True

plt.switch_backend('pdf')


def show_img(im, figsize=None, ax=None, alpha=None):
    if not ax:
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(im, alpha=alpha)
        ax.set_axis_off()
        return ax


def dice(preds, targs):
    pred = (preds > 0).float()
    return 2. * (pred * targs).sum() / (pred.sum() + targs.sum())


DATA_PATH = 'data'
TRAIN_PATH = 'train'
VAL_PATH = 'val'
IMG_PATH = 'images/0'
MASKS_PATH = 'masks/0'


class MatchedFilesDatas(FilesDataset):
    def __init__(self, fnames, y, transform, path):
        self.y = y
        assert (len(fnames) == len(y))
        super().__init__(fnames, transform, path)

    def get_y(self, i):
        return open_image(os.path.join(self.path, self.y[i]))

    def get_c(self):
        return 0


trn_names = os.listdir(Path(DATA_PATH) / Path(TRAIN_PATH) / Path(IMG_PATH))
trn_x = np.array(
    [Path(TRAIN_PATH) / Path(IMG_PATH) / f'{fi}' for fi in trn_names])
trn_y = np.array(
    [Path(TRAIN_PATH) / Path(MASKS_PATH) / f'{fi}' for fi in trn_names])
val_names = os.listdir(Path(DATA_PATH) / Path(VAL_PATH) / Path(IMG_PATH))
val_x = np.array(
    [Path(VAL_PATH) / Path(IMG_PATH) / f'{fi}' for fi in val_names])
val_y = np.array(
    [Path(VAL_PATH) / Path(MASKS_PATH) / f'{fi}' for fi in val_names])

aug_tfms = [
    RandomRotate(4, tfm_y=TfmType.CLASS),
    RandomFlip(tfm_y=TfmType.CLASS),
    RandomLighting(0.05, 0.05, tfm_y=TfmType.CLASS)
]

f = resnet34
cut, lr_cut = model_meta[f]


def get_base():
    layers = cut_model(f(True), cut)
    return nn.Sequential(*layers)


class SaveFeatures():
    features = None

    def __init__(self, m):
        self.hook = m.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.features = output

    def remove(self):
        self.hook.remove()


class UnetBlock(nn.Module):
    def __init__(self, up_in, x_in, n_out):
        super().__init__()
        up_out = x_out = n_out // 2
        self.x_conv = nn.Conv2d(
            x_in, x_out, kernel_size=3, stride=1, padding=1)
        self.tr_conv = nn.Upsample(scale_factor=2, mode='bilinear')
        self.after_tr_conv = nn.Conv2d(
            up_in, up_out, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(n_out)

    def forward(self, up_p, x_p):
        up_p = self.tr_conv(up_p)
        up_p = self.after_tr_conv(up_p)
        x_p = self.x_conv(x_p)
        cat_p = torch.cat([up_p, x_p], dim=1)
        return self.bn(F.relu(cat_p))


class Unet34(nn.Module):
    def __init__(self, rn):
        super().__init__()
        self.rn = rn
        self.sfs = [SaveFeatures(rn[i]) for i in [2, 4, 5, 6]]
        self.up1 = UnetBlock(512, 256, 256)
        self.up2 = UnetBlock(256, 128, 256)
        self.up3 = UnetBlock(256, 64, 256)
        self.up4 = UnetBlock(256, 64, 256)
        self.up5 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.after_up5 = nn.Conv2d(256, 1, 1)

    def forward(self, x):
        x = F.relu(self.rn(x))
        x = self.up1(x, self.sfs[3].features)
        x = self.up2(x, self.sfs[2].features)
        x = self.up3(x, self.sfs[1].features)
        x = self.up4(x, self.sfs[0].features)
        x = self.up5(x)
        x = self.after_up5(x)
        return x[:, 0]

    def close(self):
        for sf in self.sfs:
            sf.remove()


class UnetModel():
    def __init__(self, model, name='unet'):
        self.model, self.name = model, name

    def get_layer_groups(self, precompute):
        lgs = list(split_by_idxs(children(self.model.rn), [lr_cut]))
        return lgs + [children(self.model)[1:]]


m_base = get_base()
m = to_gpu(Unet34(m_base))
models = UnetModel(m)

sz = 128
bs = 16

tfms = tfms_from_model(
    resnet34,
    sz,
    crop_type=CropType.NO,
    tfm_y=TfmType.CLASS,
    aug_tfms=aug_tfms)

liver_state = A([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
tfms_1 = tfms_from_stats(
    liver_state,
    sz,
    crop_type=CropType.NO,
    tfm_y=TfmType.CLASS,
    aug_tfms=aug_tfms)

datasets = ImageData.get_ds(
    MatchedFilesDatas, (trn_x, trn_y), (val_x, val_y), tfms_1, path=DATA_PATH)
md = ImageData(DATA_PATH, datasets, bs, num_workers=4, classes=None)
denorm = md.trn_ds.denorm

learn = ConvLearner(md, models)
learn.opt_fn = optim.Adam
learn.crit = nn.BCEWithLogitsLoss()
learn.metrics = [accuracy_thresh(0.5), dice]

learn.freeze_to(1)

lr = 4e-2
wd = 1e-7

lrs = np.array([lr / 100, lr / 10, lr]) / 2
learn.fit(lr, 1, wds=wd, cycle_len=4, use_clr=(20, 8))

learn.save('tmp')
learn.load('tmp')

learn.unfreeze()
learn.bn_freeze(True)

learn.fit(lrs, 1, cycle_len=4, use_clr=(20, 8))
learn.save('128')

sz = 512
bs = 16

tfms = tfms_from_model(
    resnet34,
    sz,
    crop_type=CropType.NO,
    tfm_y=TfmType.CLASS,
    aug_tfms=aug_tfms)

tfms_1 = tfms_from_stats(
    liver_state,
    sz,
    crop_type=CropType.NO,
    tfm_y=TfmType.CLASS,
    aug_tfms=aug_tfms)

datasets = ImageData.get_ds(
    MatchedFilesDatas, (trn_x, trn_y), (val_x, val_y), tfms_1, path=DATA_PATH)
md = ImageData(DATA_PATH, datasets, bs, num_workers=4, classes=None)
denorm = md.trn_ds.denorm

learn = ConvLearner(md, models)
learn.opt_fn = optim.Adam
learn.crit = nn.BCEWithLogitsLoss()
learn.metrics = [accuracy_thresh(0.5), dice]

lr = 4e-3
wd = 1e-7
lrs = np.array([lr / 100, lr / 10, lr])

learn.freeze_to(1)
learn.load('128')
learn.fit(lr, 1, wds=wd, cycle_len=5, use_clr=(5, 5))
learn.save('512tmp')

learn.unfreeze()
learn.bn_freeze(True)

learn.load('512tmp')
learn.fit(lrs / 4, 1, wds=wd, cycle_len=20, use_clr=(20, 8))
learn.save('512end')