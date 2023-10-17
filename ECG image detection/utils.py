"""
Deyun Zhang, Apr 2023
"""

import os
import sys
import json
import pickle
import random
import numpy as np
import torch
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score


def train_one_epoch(model, optimizer, data_loader, device, epoch):

    model.train()

    nSamples = [1197, 1889, 415, 2079, 8608, 13609, 3116]
    normedWeights = [(sum(nSamples) / x) for x in nSamples]
    realweights = torch.FloatTensor(normedWeights).to(device)

    temperature = 5
    loss_function = torch.nn.BCELoss(weight=realweights)
    accu_loss = torch.zeros(1).to(device)
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader)
    for step, data in enumerate(data_loader):
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        sample_num += images.shape[0]

        pred = F.sigmoid(model(images) / temperature)

        loss = loss_function(pred, labels)
        loss.backward()
        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.3f}".format(epoch, accu_loss.item() / (step + 1))

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return accu_loss.item() / (step + 1)

@torch.no_grad()
def evaluate(model, data_loader, device, epoch):

    nSamples = [78, 113, 25, 132, 482, 723, 169]
    normedWeights = [(sum(nSamples) / x) for x in nSamples]
    realweights = torch.FloatTensor(normedWeights).to(device)
    loss_function = torch.nn.BCELoss(weight=realweights)
    temperature = 5

    model.eval()

    accu_loss = torch.zeros(1).to(device)

    sample_num = 0
    data_loader = tqdm(data_loader)

    predict_list = []
    label_list = []

    for step, data in enumerate(data_loader):
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        sample_num += images.shape[0]

        pred = F.sigmoid(model(images) / temperature)

        loss = loss_function(pred, labels)
        accu_loss += loss.detach()

        data_loader.desc = "[valid epoch {}] loss: {:.3f}".format(epoch, accu_loss.item() / (step + 1))
        pred_array = pred.cpu().data.numpy()
        labels_array = labels.cpu().data.numpy()

        predict_list.append(pred_array)
        label_list.append(labels_array)

    for idx in range(np.array(predict_list).shape[0]):
        sub_label = np.array(label_list)[idx, :]
        sub_predict = np.array(predict_list)[idx, :, :]
        if idx == 0:
            fin_label = sub_label
            fin_pred = sub_predict
        else:
            fin_label = np.concatenate((fin_label, sub_label), axis=0)
            fin_pred = np.concatenate((fin_pred, sub_predict), axis=0)

    valid_res = my_eval(fin_label, fin_pred)

    return valid_res, accu_loss.item() / (step + 1)

@torch.no_grad()
def test(model, data_loader, device, epoch):
    nSamples = [77, 122, 23, 149, 474, 717, 170]
    normedWeights = [(sum(nSamples) / x) for x in nSamples]
    realweights = torch.FloatTensor(normedWeights).to(device)
    loss_function = torch.nn.BCELoss(weight=realweights)
    temperature = 5

    model.eval()

    accu_loss = torch.zeros(1).to(device)

    sample_num = 0
    data_loader = tqdm(data_loader)

    predict_list = []
    label_list = []
    features_res = []

    timel = 0.

    for step, data in enumerate(data_loader):
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        sample_num += images.shape[0]

        pred = F.sigmoid(model(images) / temperature)


        stem_out = model.stem(images)
        blocks_out = model.blocks(stem_out)
        
        features0 = model.head.project_conv(blocks_out)
        features1 = model.head.avgpool(features0)
        features2 = model.head.flatten(features1)
        features_res.append(features2.cpu().data.numpy())

        loss = loss_function(pred, labels)
        accu_loss += loss.detach()

        data_loader.desc = "[test epoch {}] loss: {:.3f}".format(epoch, accu_loss.item() / (step + 1))

        pred_array = pred.cpu().data.numpy()
        labels_array = labels.cpu().data.numpy()

        predict_list.append(pred_array)
        label_list.append(labels_array)

    np.save('./labels_all.npy', np.array(label_list))
    np.save('./predict_all.npy', np.array(predict_list))
    np.save('./features_res.npy', np.array(features_res))

    for idx in range(np.array(predict_list).shape[0]):
        sub_label = np.array(label_list)[idx, :]
        sub_predict = np.array(predict_list)[idx, :, :]
        if idx == 0:
            fin_label = sub_label
            fin_pred = sub_predict
        else:
            fin_label = np.concatenate((fin_label, sub_label), axis=0)
            fin_pred = np.concatenate((fin_pred, sub_predict), axis=0)
            
    fin_pred = np.round(fin_pred, 3)
    prob_res = {'prob': fin_pred, 'label': fin_label}
    with open(f'./ext_test_pred_prob.pkl', 'wb') as f:
        pickle.dump(prob_res, f)

    test_res = my_eval(fin_label, fin_pred)

    return test_res, fin_pred, fin_label


def label_one_hot(label, n_classes):
    out_label = []
    for l in label:
        tmp_label = np.zeros(n_classes)
        tmp_label[int(l)] = 1

        out_label.append(tmp_label)
    return np.array(out_label)

def my_eval(y_true_idx, y_pred_prob):

    y_true = y_true_idx

    ret = []

    ret.append(roc_auc_score(y_true[:, 0], y_pred_prob[:, 0]))
    ret.append(roc_auc_score(y_true[:, 1], y_pred_prob[:, 1]))
    ret.append(roc_auc_score(y_true[:, 2], y_pred_prob[:, 2]))
    ret.append(roc_auc_score(y_true[:, 3], y_pred_prob[:, 3]))
    ret.append(roc_auc_score(y_true[:, 4], y_pred_prob[:, 4]))
    ret.append(roc_auc_score(y_true[:, 5], y_pred_prob[:, 5]))
    ret.append(roc_auc_score(y_true[:, 6], y_pred_prob[:, 6]))

    return np.array(ret)

def read_test_data(root: str):
    random.seed(0)
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    flower_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    flower_class.sort()

    class_indices = dict((k, v) for v, k in enumerate(flower_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    test_images_path = []
    test_images_label = []
    every_class_num = []
    supported = [".jpg", ".JPG", ".png", ".PNG"]

    for cla in flower_class:
        cla_path = os.path.join(root, cla)
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
        image_class = class_indices[cla]
        every_class_num.append(len(images))
        for img_path in images:
            test_images_path.append(img_path)
            test_images_label.append(image_class)

    print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for test.".format(len(test_images_path)))

    return test_images_path, test_images_label