from pathlib import Path
from torchvision import transforms, datasets
import torch

default_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


class ModelDataset():
    """
    Classe basica para manipular modelos
    """
    model: torch.nn.Module
    dataset: datasets.ImageFolder
    dataloader: torch.utils.data.DataLoader

    def load_dataset(self, path: str = "./data", tranform: torch.nn.Module = None):
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
        if not Path(path).exists():
            raise Exception('Dataset not found')

        if tranform is None:
            tranform = default_transform

        self.dataset = datasets.ImageFolder(root=path,
                                            transform=tranform)
        self.dataset_classes = self.dataset.classes

    def split_dataset(self, tresh_hold: float = 0.8):
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
        nDivision = [round(len(self.dataset) * tresh_hold),
                     round(len(self.dataset) * (1-tresh_hold))]
        self.dataset_splited = torch.utils.data.random_split(
            self.dataset, nDivision)

    def dataset_loader(self, batch_size: int = 4, num_workers: int = 4):
        """
        Descrição
        --------
        Carrega o Dataset e separa ele em dois grupos: de treino e validação e os transforma em bachs

        Entradas
        --------
        batch_size: int
        Tamanho de cada batch

        num_workers: int
        Quantidade de subprocessos
        """

        self.dataloaders = [torch.utils.data.DataLoader(d, batch_size=batch_size, shuffle=True, num_workers=num_workers)
                            for d in self.dataset_splited]

    def dataset_load_all(self, path: str = "./data", tranform: torch.nn.Module = None, tresh_hold: float = 0.8, batch_size: int = 4, num_workers: int = 4):
        """Vê as documentações das outras ai por favor, nunca te pedi nada"""
        self.load_dataset(path, tranform)
        self.split_dataset(tresh_hold)
        self.dataset_loader(batch_size, num_workers)
