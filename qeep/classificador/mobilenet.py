from typing import List

import torch
from qeep.classificador.model_base import ModelUtil


class MobileNetBasic(ModelUtil):
    """
    Mobilenet pre treinada do torch hub alterando a ultima camada
    """

    def __init__(
        self,
        output_size: int,
        class_names: List[str] = None,
        freeze: bool = True,
    ):
        # model = torch.load(
        #     "pytorch/vision:v0.6.0", "mobilenet_v2", pretrained=True
        # )
        model = torch.load("mobilenet.pkl")
        # if freeze:
        #     for param in model.parameters():
        #         param.requires_grad = False
        # num_ftrs = model.classifier[-1].in_features
        # model.classifier[-1] = nn.Linear(num_ftrs, output_size)
        # model.classifier.add_module("2", nn.LogSoftmax())

        self.model = model.to(self.device)
        self.class_names = class_names
