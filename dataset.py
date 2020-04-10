import os
import imgaug.augmenters as iaa
import cv2
import numpy as np
from math import ceil
import numpy.random as npr
from tqdm import tqdm
import torch
import spatial_aug, temporal_aug

def make_dataset():
    labels = {}
    with open('./labels.csv', 'r') as fr:
        for ln in fr:
            num, label = ln.split('\t')
            num = int(num)
            label = label.strip()
            try:
                labels[label].append(num)
            except:
                labels[label] = [num]
    print('Found %d labels' % len(labels))
    train_ds = Dataset(labels, 0.7)
    test_ds = Dataset(labels, -0.3)
    return train_ds, test_ds

class Dataset(object):
    def __init__(self, labels, part):
        self.resizer = iaa.Resize({'width' : 64, 'height' : 64})
        self.mean = 109.71164610001608
        self.std = 58.766042004213176
        self.samples = []
        self.targets = []
        self.batch_size = 64
        self.spatial_aug = spatial_aug.Compose([
            spatial_aug.MultiScaleRandomCrop(),
            spatial_aug.SpatialElasticDisplacement()
        ])
        my_labels = {}
        for label in labels:
            nums = labels[label]
            if part > 0:
                my_labels[label] = nums[:int(len(nums)*part)]
            else:
                my_labels[label] = nums[int(len(nums)*part):]

        for label in my_labels:
            nums = my_labels[label]
            print('Loading %d paths for label %s' % (len(nums), label))
            for _, num in tqdm(enumerate(nums)):
                fls = sorted(os.listdir('./data/%d/' % num))
                if len(fls) < 8:
                    continue
                imgs = []
                for f_name in fls:
                    img = cv2.imread('./data/%d/%s' % (num, f_name))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    sample_img = img.reshape((1, img.shape[0], img.shape[1], 1))
                    sample_img = self.resizer(images=sample_img)[0, :, :, :].reshape(1, 64, 64)
                    imgs.append(sample_img)
                for ptr in range(8, len(imgs) + 1):
                    imgs_slice = imgs[ptr - 8:ptr]
                    sample = np.stack(imgs_slice, axis=1)
                    self.samples.append(sample)
                    if label == 'No gesture':
                        self.targets.append(0)
                    else:
                        self.targets.append(1)
        self.order = np.arange(len(self.samples))

    def __getitem__(self, i):
        batch_samples = []
        batch_target = []
        if i*self.batch_size >= len(self.samples):
            raise IndexError()
        self.spatial_aug.randomize_parameters()
        for j in range(i*self.batch_size, min((i+1)*self.batch_size, len(self.samples))):
            sample = self.samples[self.order[j]].copy()
            imgs = [self.spatial_aug(sample[:, i, :, :].reshape((64, 64, 1))).reshape((1, 64, 64)) for i in range(8)]
            sample = np.stack(imgs, axis=1)
            target = self.targets[self.order[j]]
            batch_samples.append(sample)
            batch_target.append(target)
        batch = np.stack(batch_samples)
        target = np.array(batch_target)
        batch = torch.from_numpy(batch).float()
        target = torch.from_numpy(target).long()
        return (batch - self.mean)/self.std, target

    def __len__(self):
        return ceil(len(self.samples)/self.batch_size)

    def shuffle(self):
        npr.shuffle(self.order)

    def get_full(self):
        batch = np.stack(self.samples)
        target = np.array(self.targets)
        batch = torch.from_numpy(batch).float()
        target = torch.from_numpy(target).long()
        return batch/256, target
