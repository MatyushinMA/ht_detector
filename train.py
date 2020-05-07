import torch
import torch.nn as nn
from dataset import make_dataset
from sklearn.metrics import f1_score
import pickle
from mobilenet import get_model
import numpy as np
from collections import OrderedDict
from tqdm import tqdm

def filter_state_dict_by_layer(od, layer_name):
    layer_name_length = len(layer_name)
    return OrderedDict(map(lambda y : (y[layer_name_length + 1:], od[y]), filter(lambda x : x.split('.')[0] == layer_name, od)))

model = get_model(num_classes=2, sample_size=112, width_mult=0.5)
try:
    checkpoint = torch.load('./best_detector_checkpoint.pth.tar', map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint)
    print('Training from checkpoint')
except:
    checkpoint = torch.load('./pretrain/models/jester_mobilenet_0_5x_RGB_16_best.pth', map_location=torch.device('cpu'))
    weights = OrderedDict()
    for w_name in checkpoint['state_dict']:
        _w_name = '.'.join(w_name.split('.')[1:])
        weights[_w_name] = checkpoint['state_dict'][w_name]
    features_weights = filter_state_dict_by_layer(weights, 'features')
    model.features.load_state_dict(features_weights)
    print('Training from pretrain')
model.cuda()
train_ds, test_ds = make_dataset()
loss_fn = nn.CrossEntropyLoss(weight=torch.Tensor([9, 1]).float().cuda())
act_fn = nn.Softmax(dim=1)
learning_rate = 0.001
clip_value = 0.1
epochs = 100
optim = torch.optim.Adam(model.parameters(), lr=learning_rate)

train_losses = []
test_losses = []
test_f1s = []
best_precision = 0.0
"""with torch.no_grad():
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
        if len(np.unique(np_target)) > 1:
            for i, np_pred_row in enumerate(np_pred):
                print(np_target[i], np_pred_row)
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
print(TP, TN, FP, FN)
print('EPOCH: 0, TRAIN LOSS: %f, TEST LOSS: %f, PRECISION: %f, RECALL: %f, F1: %f' % (train_losses[-1], test_losses[-1], precision, recall, test_f1s[-1]))"""
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
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
    print('EPOCH: %d, TRAIN LOSS: %f, TEST LOSS: %f, PRECISION: %f, RECALL: %f, F1: %f, TP: %d, FP: %d, TN: %d, FN: %d' % (e + 1, train_losses[-1], test_losses[-1], precision, recall, test_f1s[-1], TP, FP, TN, FN))
    state_dict = model.state_dict()
    torch.save(state_dict, 'detector_checkpoint.pth.tar')
    if precision >= best_precision:
        torch.save(state_dict, 'best_precision_detector_checkpoint.pth.tar')
    if f1 == max(test_f1s):
        torch.save(state_dict, 'best_detector_checkpoint.pth.tar')
    best_precision = max(best_precision, precision)
with open('./train_losses.dat', 'wb') as fw:
    pickle.dump(train_losses, fw)
with open('./test_losses.dat', 'wb') as fw:
    pickle.dump(test_losses, fw)
with open('./test_f1s.dat', 'wb') as fw:
    pickle.dump(test_f1s, fw)
