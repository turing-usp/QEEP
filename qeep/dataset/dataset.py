from pathlib import Path
from torchvision import transforms, datasets
from typing import List
import gdown
import os
import torch
import zipfile


class PokeDataset:

    tranform: torch.nn.Module
    datasetpath: Path
    dataset: datasets.ImageFolder
    dataset_splited: List[torch.utils.data.Dataset]

    def __init__(
        self, tranforms: List[torch.nn.Module], datasetpath: str = "./data"
    ):
        """
        Descrição
        -------
        Inicializa a classe do banco de dados

        Entradas
        --------
        tranform: List[torch.nn.Module]
        Transformaçoes a serem aplicadas no dataset em formato de lista

        path: str
        Diretorio que será montado as classes na seguinte extrutura
        <path>
        ├── bulbassauro
        │   ├── imagem1.png
        │   ├── imagem2.png
        │   ├── ...
        │   └── imagemN.py
        ├── pikachu
        │   ├── imagem1.png
        │   ├── imagem2.png
        │   ├── ...
        │   └── imagemN.py
        ...

        """
        self.datasetpath = Path(datasetpath)
        self.tranform = transforms.Compose(tranforms)

    def download(
        self,
        url: str = "https://drive.google.com/uc?export=download&id=1SA7wV7BwEpNoR721aUSauFvqCTfXba1h",
    ):
        """
        Descrição
        -------
        Baixa o dataset do drive

        Entradas
        --------
        path: str
        Local em que o dataset será salvo
        """

        # Se o dataset já existe, não baixa novamente
        if self.datasetpath.exists():
            return

        datasetpath_zip = Path(self.datasetpath.name + ".zip")
        gdown.download(url, datasetpath_zip.name, quiet=False)

        with zipfile.ZipFile(datasetpath_zip, "r") as zip_ref:
            zip_ref.extractall(self.datasetpath.parent)

        os.remove(datasetpath_zip)

    def load(self):
        """
        Descrição
        --------
        Carrega o Dataset

        Entradas
        --------
        path: str
        Diretorio que será montado as classes na seguinte extrutura
        <path>
        ├── bulbassauro
        │   ├── imagem1.png
        │   ├── imagem2.png
        │   ├── ...
        │   └── imagemN.py
        ├── pikachu
        │   ├── imagem1.png
        │   ├── imagem2.png
        │   ├── ...
        │   └── imagemN.py
        ...

        tranform: torch Tranform
        Transformaçoes a serem aplicadas no dataset
        Se não for defenido será usado o default_tranform
        """
        if not self.datasetpath.exists():
            raise Exception("Dataset not found")

        self.dataset = datasets.ImageFolder(
            root=self.datasetpath, transform=self.tranform
        )
        self.dataset_classes = self.dataset.classes

    def split(self, tresh_hold: float = 0.8):
        """
        Descrição
        --------
        Separa o dataset carregado em dois grupos dividido pelo tresh_hold
        obs: precisa ter o datasetCarregado

        Entradas
        --------
        tresh_hold: float
        Porcentagem de treino em relação ao dataset original
        """
        # Se o dataset não foi inicializado, inicializa
        if self.dataset is None:
            self.load()

        nDivision = [
            round(len(self.dataset) * tresh_hold),
            round(len(self.dataset) * (1 - tresh_hold)),
        ]
        self.dataset_splited = torch.utils.data.random_split(
            self.dataset, nDivision
        )

    def loaders(
        self, batch_size: int = 4, num_workers: int = 4, shuffle: bool = True
    ) -> List[torch.utils.data.DataLoader]:
        """
        Descrição
        --------
        Mapeia os datasets para :class:`~torch.utils.data.DataLoader`

        Entradas
        --------
        batch_size: int
        Tamanho de cada batch

        num_workers: int
        Quantidade de subprocessos
        """

        return [
            torch.utils.data.DataLoader(
                d,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
            )
            for d in self.dataset_splited
        ]
