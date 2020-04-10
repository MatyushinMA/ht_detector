import torch
import torch.nn as nn
from dataset import make_dataset
from sklearn.metrics import f1_score
import pickle
from resnetl import resnetl10
import numpy as np
from collections import OrderedDict
from tqdm import tqdm

model = resnetl10(sample_size=64, sample_duration=8, num_classes=2, shortcut_type='A')
checkpoint = torch.load('./pretrain/models/egogesture_resnetl_10_Depth_8.pth', map_location=torch.device('cpu'))
weights = OrderedDict()
for w_name in checkpoint['state_dict']:
    _w_name = '.'.join(w_name.split('.')[1:])
    weights[_w_name] = checkpoint['state_dict'][w_name]
model.load_state_dict(weights)
model.cuda()
train_ds, test_ds = make_dataset()
loss_fn = nn.CrossEntropyLoss(weight=torch.Tensor([1, 3]).float().cuda())
act_fn = nn.Softmax(dim=1)
learning_rate = 0.00001
epochs = 100
optim = torch.optim.Adam(model.parameters(), lr=learning_rate)

train_losses = []
test_losses = []
test_f1s = []
with torch.no_grad():
    avg_train_loss = 0
    for batch, target in tqdm(train_ds):
        batch = batch.cuda()
        target = target.cuda()
        pred = model(batch)
        pred = act_fn(pred)
        loss = loss_fn(pred, target)
        avg_train_loss += loss.item()
    avg_train_loss /= len(train_ds)
    train_losses = [avg_train_loss]
    avg_test_loss = 0
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for batch, target in tqdm(test_ds):
        batch = batch.cuda()
        target = target.cuda()
        pred = model(batch)
        pred = act_fn(pred)
        loss = loss_fn(pred, target)
        avg_test_loss += loss.item()
        np_pred = pred.cpu().detach().numpy()
        pred_logits = np.argmax(np_pred, axis=1).reshape(-1)
        np_target = target.cpu().detach().numpy().reshape(-1)
        TP += np.sum((pred_logits == 1) & (np_target == 1))
        TN += np.sum((pred_logits == 0) & (np_target == 0))
        FP += np.sum((pred_logits == 1) & (np_target == 0))
        FN += np.sum((pred_logits == 0) & (np_target == 1))
    precision = TP/float(TP + FP)
    recall = TP/float(TP + FN)
    f1 = 2*precision*recall/(precision + recall)
    avg_test_loss /= len(test_ds)
    test_losses = [avg_test_loss]
    test_f1s = [f1]
print('EPOCH: 0, TRAIN LOSS: %f, TEST LOSS: %f, PRECISION: %f, RECALL: %f, F1: %f' % (train_losses[-1], test_losses[-1], precision, recall, test_f1s[-1]))
for e in range(epochs):
    train_ds.shuffle()
    avg_train_loss = 0
    for batch, target in tqdm(train_ds):
        optim.zero_grad()
        batch = batch.cuda()
        target = target.cuda()
        pred = model(batch)
        pred = act_fn(pred)
        loss = loss_fn(pred, target)
        avg_train_loss += loss.item()
        loss.backward()
        optim.step()
    avg_train_loss /= len(train_ds)
    train_losses.append(avg_train_loss)
    avg_test_loss = 0
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    with torch.no_grad():
        for eval_batch, eval_target in tqdm(test_ds):
            eval_batch = eval_batch.cuda()
            eval_target = eval_target.cuda()
            pred = model(eval_batch)
            pred = act_fn(pred)
            loss = loss_fn(pred, eval_target)
            avg_test_loss += loss.item()
            np_pred = pred.cpu().detach().numpy()
            pred_logits = np.argmax(np_pred, axis=1).reshape(-1)
            np_target = eval_target.cpu().detach().numpy().reshape(-1)
            TP += np.sum((pred_logits == 1) & (np_target == 1))
            TN += np.sum((pred_logits == 0) & (np_target == 0))
            FP += np.sum((pred_logits == 1) & (np_target == 0))
            FN += np.sum((pred_logits == 0) & (np_target == 1))
    precision = TP/float(TP + FP)
    recall = TP/float(TP + FN)
    f1 = 2*precision*recall/(precision + recall)
    avg_test_loss /= len(test_ds)
    test_losses.append(avg_test_loss)
    test_f1s.append(f1)
    print('EPOCH: %d, TRAIN LOSS: %f, TEST LOSS: %f, PRECISION: %f, RECALL: %f, F1: %f' % (e + 1, train_losses[-1], test_losses[-1], precision, recall, test_f1s[-1]))
with open('./train_losses.dat', 'wb') as fw:
    pickle.dump(train_losses, fw)
with open('./test_losses.dat', 'wb') as fw:
    pickle.dump(test_losses, fw)
with open('./test_f1s.dat', 'wb') as fw:
    pickle.dump(test_f1s, fw)
    state_dict = model.state_dict()
    torch.save(state_dict, 'detector_checkpoint.pth.tar')
    if f1 == max(test_f1s):
        torch.save(state_dict, 'best_detector_checkpoint.pth.tar')
