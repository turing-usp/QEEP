#! /usr/bin/python

import argparse

import torch.nn as nn
import torch.optim as optim
from torchvision import transforms

from qeep.classificador.mobilenet import MobileNetBasic  # noqa: I900
from qeep.classificador.model_base import ModelUtil  # noqa: I900
from qeep.classificador.pokenet import PokeMobileNet  # noqa: I900
from qeep.dataset.dataset import PokeDataset  # noqa: I900


def train(  # noqa: PLR0913
    model: ModelUtil,
    model_name: str,
    tresh_hold: float,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    dataset_path: str,
    dataset_drive_id: str,
    optimizer_learning_rate: float,
    optimizer_momentum: float,
    scheduler_step_size: int,
    scheduler_gamma: float,
    epochs: int,
    verbose: bool,
):
    """ Treina um modelo"""
    if verbose:
        model.show()

    trasnform_augumentation = model.transforms + [
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
    ]
    dataset = PokeDataset(trasnform_augumentation, dataset_path)
    if dataset_drive_id:
        dataset.download(dataset_drive_id)
    [train_dl, val_dl, *_] = dataset.loaders(
        batch_size,
        num_workers,
        shuffle,
        tresh_hold=tresh_hold,
    )

    criterion = nn.NLLLoss()
    optimizer = optim.SGD(
        model.model.parameters(),
        lr=optimizer_learning_rate,
        momentum=optimizer_momentum,
    )
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=scheduler_step_size,
        gamma=scheduler_gamma,
    )

    model.train(
        criterion,
        optimizer,
        scheduler,
        train_dl,
        val_dl,
        epochs=epochs,
    )

    model.save(model_name)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Treina uma rede")

    parser.add_argument(
        "--model",
        "-m",
        dest="model",
        default="PokeMobilenet",
        type=str,
        help="Rede que será usada",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        dest="verbose",
        default=True,
        type=bool,
        help="Mostra prints",
    )

    parser.add_argument(
        "--dataset-path",
        "-d",
        dest="dataset_path",
        default="./data",
        type=str,
        help="Local em que o dataset será carregado",
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

    args = parser.parse_args()

    # lazy load
    models = {
        "MobileNetBasicl": lambda: MobileNetBasic(151),
        "PokeMobileNet": lambda: PokeMobileNet(),  # noqa: PLW0108
    }

    train(
        models[args.model](),
        args.model + ".pt",
        args.tresh_hold,
        args.batch_size,
        args.shuffle,
        args.num_workers,
        args.dataset_path,
        args.dataset_drive_id,
        args.optimizer_learning_rate,
        args.optimizer_momentum,
        args.scheduler_step_size,
        args.scheduler_gamma,
        args.epochs,
        args.verbose,
    )
