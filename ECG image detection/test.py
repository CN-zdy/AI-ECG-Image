"""
Deyun Zhang, Apr 2023
"""

import os
import argparse
import torch
from torchvision import transforms
import pickle
from model import efficientnetv2_s as create_model
from my_dataset import MyDataSet
from utils import  test, read_test_data


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    test_images_path, test_images_label = read_test_data(args.test_path)

    img_size = {"s": [300, 384],
                "m": [384, 480],
                "l": [384, 480],
                "xl": [384, 512]}
    num_model = "s"

    data_transform = {
        "test": transforms.Compose([transforms.Resize(img_size[num_model][1]),
                                   transforms.CenterCrop(img_size[num_model][1]),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])}

    test_dataset = MyDataSet(images_path=test_images_path,
                            images_class=test_images_label,
                            transform=data_transform["test"])

    batch_size = args.batch_size

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=16,
                                             drop_last=True,
                                             collate_fn=test_dataset.collate_fn)

    model = create_model(num_classes=2).to(device)

    # load model weights
    model_weight_path = "./weights_ext/"
    models = os.listdir(model_weight_path)
    for model_weight in models:
        print(model_weight)
        model.load_state_dict(torch.load(model_weight_path+model_weight, map_location=device))
        model.eval()
        with torch.no_grad():
            for epoch in range(args.epochs):
                # test
                test_res, pred_array, label_array = test(model=model,
                                         data_loader=test_loader,
                                         device=device,
                                         epoch=epoch)

                prob_res = {'prob':pred_array, 'label':label_array}
                with open(f'./test_pred_prob.pkl', 'wb') as f:
                    pickle.dump(prob_res, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=7)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lrf', type=float, default=0.01)

    parser.add_argument('--test_path', type=str, default="test image path")

    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)
