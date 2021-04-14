from typing import List
from torch.nn import Module
from qeep.classificador.model_base import ModelUtil


class PokeMobileNet(ModelUtil):
    """
    Mobilenet treinada com o Pokedataset
    """

    model: Module

    def __init__(
        self,
        class_names: List[str] = None,
    ):
        self.load("/tmp/pokedataset.pt", "1vFxsEutkyon-hDYbpL1opTcPNwfXy_yO")

        self.model = self.model.to(self.device)
        self.class_names = class_names
