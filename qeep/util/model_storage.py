import gdown
import torch
import torch.nn as nn
import os
from typing import Type
from pathlib import Path
from torchvision import transforms, datasets

default_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


class ModelStorage():
    """
    Classe basica para manipular modelos
    """
    name: str
    device: str
    model: torch.nn.Module
    dataloader: torch.utils.data.DataLoader

    def __init__(self, name: str = "model"):
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    def show(self):
        print(self.model.eval())

    def loadModel(self, file: str = "weights.pkl", drive: bool = True, url: str = "https://drive.google.com/uc?export=download&id=1yC0qK0gVX5sc6GTpBPBupH3TSFssLkqS") -> nn.Module:
        """
        Descrição
        --------
        Carrega uma rede a ser utilizada como modelo

        Requisitos
        --------
        Precisa ter a variavel model inicializada

        Entradas
        --------   
        file: (Path)
        Caso drive = False, representa o caminho para o
        arquivo onde o modelo será carregado.
        Caso drive = True, representa o caminho para onde
        o modelo será salvo.

        drive: (Bool)
        Define se o modelos será carregado do Google Drive 
        ou localmente

        url: (Url)
        Endereço de onde o arquivo deverá ser baixado, caso
        drive = True.

        """
        if drive:
            try:
                gdown.download(url, file, quiet=False)
            except Exception as err:
                print(str(err))
                print("Rede pré treinada não pode ser baixada.")

        st = torch.load(file, map_location=self.device)
        self.model.load_state_dict(st)

    def saveModel(self, filename: str, path: str = "") -> Type[None]:
        """
        Salva um PyTorch model, no path e nome desejados, com extensão .pkl
        ---------------
        Argumentos:
            - model: PyTorch model (torch.nn.Module) 
            - path: string contendo o path para salvar o modelo
            - filename: string contendo o nome do arquivo salvo
        """
        p = os.path.join(path, filename)
        torch.save(self.model.state_dict(), p)
