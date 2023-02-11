import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from src.dataset import CocoDataset, Resizer, Normalizer, Augmenter, collater
from src.model import EfficientDet
from tensorboardX import SummaryWriter
import shutil
import numpy as np
from tqdm.autonotebook import tqdm
from src.config import device


def get_args():
    parser = argparse.ArgumentParser(
        "EfficientDet")
    parser.add_argument("--image_size", type=int, default=512, help="The common width and height for all images")
    parser.add_argument("--batch_size", type=int, default=8, help="The number of images per batch")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument('--alpha', type=float, default=0.25)
    parser.add_argument('--gamma', type=float, default=1.5)
    parser.add_argument("--num_epochs", type=int, default=500)
    parser.add_argument("--test_interval", type=int, default=1, help="Number of epoches between testing phases")
    parser.add_argument("--es_min_delta", type=float, default=0.0,
                        help="Early stopping's parameter: minimum change loss to qualify as an improvement")
    parser.add_argument("--es_patience", type=int, default=0,
                        help="Early stopping's parameter: number of epochs with no improvement after which training will be stopped. Set to 0 to disable this technique.")
    parser.add_argument("--data_path", type=str, default="data/COCO", help="the root folder of dataset")
    parser.add_argument("--log_path", type=str, default="tensorboard/efficientdet_coco")
    parser.add_argument("--saved_path", type=str, default="trained_models")
    parser.add_argument("--pretrained_model", type=str, default="trained_models/efficientdet_coco.pth")
    parser.add_argument("--last_train_epoch", type=str, default="trained_models/last_train_epoch.txt")

    args = parser.parse_args()
    return args


def save_onnx(opt):
    num_gpus = 1
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
    torch.manual_seed(123)

    training_params = {"batch_size": opt.batch_size * num_gpus,
                       "shuffle": True,
                       "drop_last": True,
                       "collate_fn": collater,
                       "num_workers": 8}

    training_set = CocoDataset(root_dir=opt.data_path, set="train2017",
                               transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))

    model = EfficientDet(num_classes=training_set.num_classes())


    if os.path.isdir(opt.log_path):
        shutil.rmtree(opt.log_path)
    os.makedirs(opt.log_path)

    if not os.path.isdir(opt.saved_path):
        os.makedirs(opt.saved_path)


    if os.path.isfile(opt.pretrained_model):
      model.load_state_dict(torch.load(opt.pretrained_model, map_location=device))
      print("Training Checkpoint Obtained")

    if torch.cuda.is_available():
        model = nn.DataParallel(model)
    model.to(device=device)

    dummy_input = torch.rand(opt.batch_size, 3, 512, 512)
    dummy_input = dummy_input.to(device=device)

    if isinstance(model, nn.DataParallel):
        print("This runs")
        model.module.backbone_net.model.set_swish(memory_efficient=False)

        torch.onnx.export(model.module, dummy_input,
                          os.path.join(opt.saved_path, "efficientdet_coco.onnx"),
                          verbose=False)
        model.module.backbone_net.model.set_swish(memory_efficient=True)
    else:
        print("That runs")
        model.backbone_net.model.set_swish(memory_efficient=False)

        torch.onnx.export(model, dummy_input,
                        os.path.join(opt.saved_path, "efficientdet_coco.onnx"),
                          verbose=False)
        model.backbone_net.model.set_swish(memory_efficient=True)

if __name__ == "__main__":
    opt = get_args()
    save_onnx(opt)
