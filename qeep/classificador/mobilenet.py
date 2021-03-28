"""
    MobileNet
"""

import json
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from ..util.model import ModelUtil
from ..dataset.dataset import PokeDataset


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


def _main(args: argparse.Namespace):
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
        dataset.download(args.dataset_url)
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
        print(mobilenet.predict(args.img_path))

    if args.save:
        mobilenet.save(args.save)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Treina na mobilet")

    parser.add_argument(
        "--show",
        dest="show",
        default=False,
        type=bool,
        help="Mostra a rede",
    )

    parser.add_argument(
        "--load",
        dest="load",
        default=False,
        type=bool,
        help="Carrega os pesos do modelo",
    )

    parser.add_argument(
        "--input",
        dest="img_path",
        type=str,
        help="Imagem que servirá de entrada para o modelo",
    )

    parser.add_argument(
        "--classes",
        dest="class_names_path",
        default="./classes.json",
        type=str,
        help="Path do json com as classes",
    )

    parser.add_argument(
        "--train",
        dest="is_train",
        default=False,
        type=bool,
        help="Define que o modelo deve ser treinado",
    )

    parser.add_argument(
        "--dataset-path",
        dest="dataset_path",
        default="./data",
        type=str,
        help="Local em que o dataset será carregado, caso não sejá encontrado será baixado",
    )

    parser.add_argument(
        "--dataset-url",
        dest="dataset_url",
        default="https://drive.google.com/uc?export=download&id=1SA7wV7BwEpNoR721aUSauFvqCTfXba1h",
        type=str,
        help="URL do drive na qual o dataset será baixado",
    )

    parser.add_argument(
        "--dataloader-batch-size",
        dest="dataloader_batch_size",
        default=4,
        type=int,
        help="Bathsize do dataloader",
    )

    parser.add_argument(
        "--dataloader-num-workers",
        dest="dataloader_num_workers",
        default=4,
        type=int,
        help="Numero de workrs do dataloader",
    )

    parser.add_argument(
        "--dataloader-shuffle",
        dest="dataloader_shuffle",
        default=True,
        type=bool,
        help="Embraralhar dataloader",
    )

    parser.add_argument(
        "--dataset-tresh-hold",
        dest="dataset_tresh_hold",
        default=0.8,
        type=float,
        help="Porcentagem do dataset usado para treino",
    )

    parser.add_argument(
        "--optimizer-learning-rate",
        dest="optimizer_learning_rate",
        default=0.001,
        type=float,
        help="Learning rate do otimizador",
    )

    parser.add_argument(
        "--optimizer-momentum",
        dest="optimizer_momentum",
        default=0.9,
        type=float,
        help="Momentum do otimizador",
    )

    parser.add_argument(
        "--scheduler-step-size",
        dest="scheduler_step_size",
        default=7,
        type=int,
        help="Step size do schedule",
    )

    parser.add_argument(
        "--scheduler-gamma",
        dest="scheduler_gamma",
        default=0.1,
        type=float,
        help="Step size do schedule",
    )

    parser.add_argument(
        "--epochs",
        dest="epochs",
        default=25,
        type=int,
        help="Numero de epocas treinadas",
    )

    parser.add_argument(
        "--save",
        dest="save",
        default="mobilenet_weight.pkl",
        type=str,
        help="Para salvar o modelo",
    )

    parse_args = parser.parse_args()

    if (not parse_args.is_train) and (parse_args.img_path is None):
        print("input image is required to run")
        sys.exit(1)

    _main(parse_args)
