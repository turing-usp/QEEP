"""
    MobileNet
"""

import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from util.model import ModelUtil
from dataset.dataset import PokeDataset


class MobileNet(ModelUtil):
    """
    MobileNet
    """

    def __init__(
        self, output_size: int, class_names: [str] = None, freeze: bool = True
    ):
        model = torch.hub.load(
            "pytorch/vision:v0.6.0", "mobilenet_v2", pretrained=True
        )
        if freeze:
            for param in model.parameters():
                param.requires_grad = False
        num_ftrs = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(num_ftrs, output_size)
        model.classifier.add_module("2", nn.LogSoftmax())

        self.model = model.to(self.device)
        self.class_names = class_names


def run(args: argparse.Namespace):
    mobilenet = MobileNet(151)

    if args.load:
        mobilenet.load("mobilenet_weights.pkl")

    if args.show:
        mobilenet.show()

    if args.is_train:
        trasnform_augumentation = mobilenet.transforms + [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
        ]
        dataset = PokeDataset(trasnform_augumentation, args.dataset_path)
        dataset.download(args.dataset_url_id)
        dataset.load()
        mobilenet.class_names = dataset.dataset_classes
        dataset.split(args.dataset_tresh_hold)
        [train_dl, val_dl, *_] = dataset.loaders(
            args.dataloader_batch_size,
            args.dataloader_num_workers,
            args.dataloader_shuffle,
        )

        criterion = nn.NLLLoss()
        optimizer = optim.SGD(
            mobilenet.model.parameters(),
            lr=args.optimizer_learning_rate,
            momentum=args.optimizer_momentum,
        )
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=args.scheduler_step_size,
            gamma=args.scheduler_gamma,
        )

        mobilenet.train(
            criterion,
            optimizer,
            scheduler,
            train_dl,
            val_dl,
            epochs=args.epochs,
        )
    else:
        with open(args.class_names_path) as classes_file:
            mobilenet.class_names = json.load(classes_file)
        print(mobilenet.predict(args.img_path)[1])

    if args.save:
        mobilenet.save(args.save)
