"""
Deyun Zhang, Apr 2023
"""

import os
import argparse
import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms
from model import efficientnetv2_s as create_model_s
from my_dataset import MyDataSet
from utils import train_one_epoch, evaluate, read_data

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if os.path.exists(f"./{args.save_path}/weights") is False:
        os.makedirs(f"./{args.save_path}/weights")

    train_images_path, train_images_label = read_data(args.train_path)
    val_images_path, val_images_label = read_data(args.valid_path)
    test_images_path, test_images_label = read_data(args.test_path)

    img_size = {"s": [300, 384],
                "m": [384, 480],
                "l": [384, 480],
                "xl": [384, 512]}
    num_model = "s"

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

    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=data_transform["train"])
  
    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform["val"])

    test_dataset = MyDataSet(images_path=test_images_path,
                            images_class=test_images_label,
                            transform=data_transform["test"])

    batch_size = args.batch_size

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=16,
                                               drop_last=True,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             pin_memory=True,
                                             num_workers=16,
                                             drop_last=True,
                                             collate_fn=val_dataset.collate_fn)

    model = create_model_s(num_classes=args.num_classes).to(device)
    if args.weights != "":
        if os.path.exists(args.weights):
            weights_dict = torch.load(args.weights, map_location=device)

            load_weights_dict = {k: v for k, v in weights_dict.items()
                                 if model.state_dict()[k].numel() == v.numel()}
            print(model.load_state_dict(load_weights_dict, strict=False))
        else:
            raise FileNotFoundError("not found weights file: {}".format(args.weights))

    if args.freeze_layers:
        for name, para in model.named_parameters():
            if "head" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))

    model.to(device)

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(pg, lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=4, factor=0.3, mode='max')\

    v_roc = 0.
    early_stop_lr = 1e-6

    for epoch in range(args.epochs):
        # train
        train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch)

        # validate
        val_res, val_loss, val_acc = evaluate(model=model,
                                     data_loader=val_loader,
                                     device=device,
                                     epoch=epoch)

        current_roc = np.mean(val_res[:2])
        scheduler.step(current_roc)

        if current_roc > v_roc:
            v_roc = current_roc
            print("Save Model current_roc is {}" .format(current_roc))
            torch.save(model.state_dict(), f"{args.save_path}/weights/roc_model-{epoch}.pth",_use_new_zipfile_serialization=False)

        current_lr = optimizer.state_dict()['param_groups'][0]['lr']
        if current_lr < early_stop_lr:
            print(f"Early stop ...")
            break



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=int(1e5))
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lrf', type=float, default=0.01)

    parser.add_argument('--save_path', type=str, default="/data1/1shared/deyun/PSAF/results_t0/")
    parser.add_argument('--train_path', type=str, default="/data1/1shared/deyun/PSAF/data_preprocess/tvt/train/")
    parser.add_argument('--valid_path', type=str, default="/data1/1shared/deyun/PSAF/data_preprocess/tvt/valid/")
    parser.add_argument('--test_path', type=str, default="/data1/1shared/deyun/PSAF/data_preprocess/tvt/test/")

    parser.add_argument('--weights', type=str, default='./pre_efficientnetv2-s.pth',
                        help='initial weights path')
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:2', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)
