"""
Deyun Zhang, Apr 2023
"""

import os
import sys
import json
import random
import numpy as np
import torch
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score

def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()

    nSamples = [2437, 2833]
    normedWeights = [1 - (x / sum(nSamples)) for x in nSamples]
    normedWeights = torch.FloatTensor(normedWeights).to(device)
    loss_function = torch.nn.CrossEntropyLoss(weight=normedWeights)

    accu_loss = torch.zeros(1).to(device)
    accu_num = torch.zeros(1).to(device)
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        loss.backward()
        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


@torch.no_grad()
def evaluate(model, data_loader, device, epoch):

    nSamples = [134, 172]
    normedWeights = [1 - (x / sum(nSamples)) for x in nSamples]
    normedWeights = torch.FloatTensor(normedWeights).to(device)
    loss_function = torch.nn.CrossEntropyLoss(weight=normedWeights)

    model.eval()

    accu_num = torch.zeros(1).to(device)
    accu_loss = torch.zeros(1).to(device)

    sample_num = 0
    data_loader = tqdm(data_loader)

    predict_list = []
    label_list = []

    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        accu_loss += loss.detach()

        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)
        pred_array = F.softmax(pred, dim=1).cpu().data.numpy()
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

    return valid_res, accu_loss.item() / (step + 1), accu_num.item() / sample_num

@torch.no_grad()
def test(model, data_loader, device, epoch):

    nSamples = [128, 162]
    normedWeights = [1 - (x / sum(nSamples)) for x in nSamples]
    normedWeights = torch.FloatTensor(normedWeights).to(device)
    loss_function = torch.nn.CrossEntropyLoss(weight=normedWeights)

    model.eval()

    accu_num = torch.zeros(1).to(device)
    accu_loss = torch.zeros(1).to(device)
    sample_num = 0
    data_loader = tqdm(data_loader)

    predict_list = []
    label_list = []
    pred_list = []
    features_res = []

    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))

        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        stem_out = model.stem(images.to(device))
        blocks_out = model.blocks(stem_out)
        
        features0 = model.head.project_conv(blocks_out)
        features1 = model.head.avgpool(features0)
        features2 = model.head.flatten(features1)
        features_res.append(features2.cpu().data.numpy())

        loss = loss_function(pred, labels.to(device))
        accu_loss += loss

        data_loader.desc = "[test epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

        pred_array = F.softmax(pred, dim=1).cpu().data.numpy()
        labels_array = labels.cpu().data.numpy()

        predict_list.append(pred_array)
        label_list.append(labels_array)
        pred_list.append(pred.cpu().data.numpy())

    np.save('./pred.npy', np.array(predict_list))
    np.save('./label.npy', np.array(label_list))
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

    test_res = my_eval(fin_label, fin_pred)

    return test_res, fin_pred, fin_label


def label_one_hot(label, n_classes):
    out_label = []
    for l in label:
        tmp_label = np.zeros(n_classes)

        tmp_label[int(l)] = 1

        out_label.append(tmp_label)
    return np.array(out_label)

def my_eval(y_true_idx, y_pred_prob, n_classes=2):

    y_true = label_one_hot(y_true_idx, n_classes)

    y_pred = np.zeros_like(y_pred_prob)
    y_pred[np.arange(y_pred_prob.shape[0]), np.argmax(y_pred_prob, axis=1)] = 1

    ret = []

    ret.append(roc_auc_score(y_true[:, 0], y_pred_prob[:, 0]))
    ret.append(roc_auc_score(y_true[:, 1], y_pred_prob[:, 1]))

    ret.append(accuracy_score(y_true, y_pred))
    ret.append(recall_score(y_true, y_pred, average='weighted'))
    ret.append(f1_score(y_true, y_pred, average='weighted'))

    return np.array(ret)

def read_data(root: str):
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