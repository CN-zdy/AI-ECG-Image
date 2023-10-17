"""
Deyun Zhang, Apr 2023
"""

import os
import argparse
import numpy as np
import torch
import warnings
import torch.optim as optim
from torchvision import transforms
import adabound
from model import efficientnetv2_s as create_model_s
from my_dataset import MyDataSet
from utils import train_one_epoch, evaluate
warnings.filterwarnings("ignore")

def read_file(str1, str2):
    img_list = []
    label_list = []
    files = os.listdir(str1)
    for file in files:
        filename = file[:-len('.jpg')]

        img_str = str1 + file
        label_str = str2 + filename+'.npy'

        img_list.append(img_str)
        label_list .append(label_str)

    return img_list, label_list

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # generate save path
    if os.path.exists("./weight") is False:
        os.makedirs("./weight")
    
    # get filename list from path
    train_images_path, train_images_label = read_file('train images path', 'train labels path')
    valid_images_path, valid_images_label = read_file('valid images path', 'valid labels path')
    test_images_path, test_images_label = read_file('test images path', 'test labels path')

    # weighted of loss function
    nSamples = [78, 113, 25, 132, 482, 723, 169]
    normedWeights = [(sum(nSamples) / x) for x in nSamples]
    realweights = [(x / sum(normedWeights)) for x in normedWeights]

    # set image size
    img_size = {"s": [300, 384],
                "m": [384, 480],
                "l": [384, 480],
                "xl": [384, 512]}
    num_model = "s"

    # set transform
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(img_size[num_model][0]),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
        "val": transforms.Compose([transforms.Resize(img_size[num_model][1]),
                                   transforms.CenterCrop(img_size[num_model][1]),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
        "test": transforms.Compose([transforms.Resize(img_size[num_model][1]),
                                   transforms.CenterCrop(img_size[num_model][1]),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])}
    
    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 16])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    # dataloader
    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=data_transform["train"])
    val_dataset = MyDataSet(images_path=valid_images_path,
                            images_class=valid_images_label,
                            transform=data_transform["val"])
    test_dataset = MyDataSet(images_path=test_images_path,
                            images_class=test_images_label,
                            transform=data_transform["test"])

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               drop_last=True,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             pin_memory=True,
                                             num_workers=nw,
                                             drop_last=True,
                                             collate_fn=val_dataset.collate_fn)

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                             batch_size=1,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw*batch_size,
                                             drop_last=False,
                                             collate_fn=test_dataset.collate_fn)


    # load pretrain model
    model = create_model_s(num_classes=args.num_classes).to(device)
    if args.weights != "":
        if os.path.exists(args.weights):
            weights_dict = torch.load(args.weights, map_location=device)

            load_weights_dict = {k: v for k, v in weights_dict.items()
                                 if model.state_dict()[k].numel() == v.numel()}
            print(model.load_state_dict(load_weights_dict, strict=False))
        else:
            raise FileNotFoundError("not found weights file: {}".format(args.weights))

    # freeze layers
    if args.freeze_layers:
        for name, para in model.named_parameters():
            if "head" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))

    model.to(device)

    # optimizer and lr scheduler
    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = adabound.AdaBound(pg, lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=4, factor=0.5, mode='max')

    v_roc = 0.
    early_stop_lr = 1e-6

    for epoch in range(args.epochs):
        # train
        train_loss = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch)

        # validate
        val_res, val_loss = evaluate(model=model,
                                     data_loader=val_loader,
                                     device=device,
                                     epoch=epoch)

        current_roc = np.sum(realweights * val_res[:])
        scheduler.step(current_roc)

        if current_roc > v_roc:
            v_roc = current_roc
            print("Save Model current_roc is {}" .format(current_roc))
            torch.save(model.state_dict(), "./weights_s/roc_model_{}.pth".format(epoch),_use_new_zipfile_serialization=False)

        current_lr = optimizer.state_dict()['param_groups'][0]['lr']

        if current_lr < early_stop_lr:
            print(f"Early stop ...")
            break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=7)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lrf', type=float, default=0.01)

    parser.add_argument('--data-path', type=str,
                        default="/data1/1shared/deyun/CinC2020_single/classification_exp/tv/")
    parser.add_argument('--test_path', type=str, default="/data1/1shared/deyun/CinC2020_single/classification_exp/test/")


    parser.add_argument('--weights', type=str, default='pre_efficientnetv2-s.pth',
                        help='initial weights path')
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:1', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)
