import torch
import torch.nn as nn
from dataset import make_dataset
from sklearn.metrics import f1_score
import pickle
from mobilenet import get_model
import numpy as np
from collections import OrderedDict
from tqdm import tqdm
import cv2
import os

model = get_model(num_classes=2, sample_size=112, width_mult=0.5)
checkpoint = torch.load('./best_precision_detector_checkpoint.pth.tar', map_location=torch.device('cpu'))
model.load_state_dict(checkpoint)
model.cuda()
train_ds, test_ds = make_dataset()
act_fn = nn.Softmax(dim=1)
mean = 109.71164610001608
std = 58.766042004213176
with torch.no_grad():
    for k, (batch, target) in tqdm(enumerate(train_ds)):
        batch = batch.cuda()
        pred = model(batch)
        pred = act_fn(pred).cpu().numpy()
        pred_values = np.argmax(pred, axis=1).reshape(-1)
        target = target.numpy().reshape(-1)
        if len(np.unique(target)) > 1:
            for i, pred_val in enumerate(pred_values):
                if target[i] == pred_val:
                    os.system('mkdir /tmp/gt/%d_%d_%d' % (k, i, pred_val))
                    sample = batch[i].cpu().numpy()
                    for j in range(sample.shape[1]):
                        img = (sample[:, j, :, :]*std + mean).reshape((112, 112, 3))
                        cv2.imwrite('/tmp/gt/%d_%d_%d/%d.jpg' % (k, i, pred_val, j), img)
                    with open('/tmp/gt/%d_%d_%d/pred.out' % (k, i, pred_val), 'w') as fw:
                        fw.write(str(pred[i]))
