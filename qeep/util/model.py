from pathlib import Path
from torch.optim import lr_scheduler
from torchvision import transforms, datasets
from typing import Type, List
import copy
import gdown
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import zipfile

default_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


class ModelUtil():
    """
    Classe basica para manipular, treinar, e carregar modelos
    """
    model: torch.nn.Module
    dataset: datasets.ImageFolder
    dataloaders: List[torch.utils.data.DataLoader]

    def __call__(self, x):
        self.model(x)

    @property
    def device(self, name: str = "model") -> str:
        return 'cuda:0' if torch.cuda.is_available() else 'cpu'

    def trainModel(self, criterion=None, optimizer=None, scheduler: optim.lr_scheduler.StepLR = None,
                   epochs: int = 25, learning_rate: float = 0.001):
        if criterion is None:
            criterion = nn.CrossEntropyLoss()
        if optimizer is None:
            optimizer = optim.SGD(self.model.parameters(),
                                  lr=learning_rate, momentum=0.9)
        if scheduler is None:
            scheduler = optim.lr_scheduler.StepLR(
                optimizer, step_size=7, gamma=0.1)

        for epoch in range(epochs):
            print('Epoch {}/{}'.format(epoch, epochs - 1))
            print('-' * 10)

            # Train
            self.model.train()
            running_loss = 0.0
            running_corrects = 0
            # Iterate over data.
            for inputs, labels in self.dataloaders[0]:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(True):
                    outputs = self.model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                scheduler.step()

            epoch_loss = running_loss / len(self.dataloaders[0])
            epoch_acc = running_corrects.double() / len(self.dataloaders[0])

            print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            print(f'Validation Accuracy: {self.modelAccuracy()}')
            print()

    def modelAccuracy(self, dataloader: torch.utils.data.DataLoader = None):
        """
        Descrição
        --------
        Função que calcula a acurácia de um modelo dado um dataloader e o device.

        model_dataloader: (torch.Tensor)
        Dataloader que contém o conjunto de imagens e rótulos.

        Saídas
        ------
        accuracy: (float)
        Acurácia do modelo para o conjunto de dados do dataloader.
        """
        corrects = 0
        total = 0

        if dataloader is None:
            dataloader = self.dataloaders[-1]

        with torch.no_grad():
            for batch in dataloader:
                images, labels = batch
                outputs = self.model(images.to(self.device))
                _, predicts = torch.max(outputs.data, 1)
                total += labels.size(0)
                corrects += (predicts.cpu() == labels).sum().item()

        return 100 * (corrects / total)

    def download_dataset(self, path: str = "./data", url: str = "https://drive.google.com/uc?export=download&id=1SA7wV7BwEpNoR721aUSauFvqCTfXba1h"):
        """
        Descrição
        -------
        Baixa o dataset do drive

        Local em que o dataset será salvo
        """
        p = Path(path + ".zip")
        pzip = Path(p.name + ".zip")

        if p.exists():
            return

        print("dowload from", url)
        try:
            gdown.download(url, pzip.name, quiet=False)
        except Exception as err:
            print("Arquivo não encontrado")
            print(str(err))
            return

        with zipfile.ZipFile(pzip, 'r') as zip_ref:
            zip_ref.extractall(p.parent)

        os.remove(pzip)

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
        self.download_dataset(path)
        self.load_dataset(path, tranform)
        self.split_dataset(tresh_hold)
        self.dataset_loader(batch_size, num_workers)

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
