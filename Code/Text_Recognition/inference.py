# -*- coding: utf-8 -*-

import torch
from torch.autograd import Variable
import utils
import dataset
from PIL import Image

import models.crnn as crnn


img_path = './data/demo.png'

# initialize model
model_path = './expr/final.pth'
alphabet = "0123456789aáàâäbcdeéèêfghiìîjklmnoóòôöpqrstuúùûüvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,!?:_@#&+-/%€()'"
alphabet = utils.convertFromChars(alphabet)

# load the pre-trained CRNN model
print("loading CRNN model...")
model = torch.load(model_path)

converter = utils.strLabelConverter(alphabet)
transformer = dataset.resizeNormalize((100, 32))

image = Image.open(img_path).convert('L')
image = transformer(image)
if torch.cuda.is_available():
    image = image.cuda()
image = image.view(1, *image.size())
image = Variable(image)

model.eval()
preds = model(image)

_, preds = preds.max(2)
preds = preds.transpose(1, 0).contiguous().view(-1)

preds_size = Variable(torch.IntTensor([preds.size(0)]))
raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
sim_pred = utils.convertToChars(sim_pred)
print('Prediction: %-20s' % (sim_pred))