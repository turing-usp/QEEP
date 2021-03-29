"""
    Basic model funcitons
"""

from typing import Type, List, Union
import time
from pathlib import Path
import copy
from PIL import Image
from torchvision import transforms
import gdown
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

DRIVE_URL = "https://drive.google.com/uc?export=download&id="


class ModelUtil:
    """
    Classe basica para manipular, treinar, e carregar modelos
    """

    model: torch.nn.Module
    class_names: List[str]

    def __call__(self, value):
        """ Allow self(x) name """
        self.model(value)

    def show(self):
        """ Print model layers """
        print(self.model.eval())

    @property
    def device(self) -> str:
        """ Try use devices """
        return "cuda:0" if torch.cuda.is_available() else "cpu"

    @property
    def transforms(self) -> List[torch.nn.Module]:
        """ Basic transforms """
        return [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]

    def train(  # noqa: PLR0914, PLR0913
        self,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler.StepLR,
        train_dataloader: torch.utils.data.DataLoader,
        val_dataloader: torch.utils.data.DataLoader,
        epochs: int = 25,
    ):
        """ Train function with gradient descendent """

        since = time.time()

        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0

        dataloaders = {"train": train_dataloader, "val": val_dataloader}

        for epoch in range(epochs):
            print(f"Epoch {epoch}/{epochs - 1}")
            print("-" * 10)

            # Each epoch has a training and validation phase
            for phase in ["train", "val"]:
                if phase == "train":
                    self.model.train()  # Set model to training mode
                else:
                    self.model.eval()  # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == "train"):
                        outputs = self.model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == "train":
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                if phase == "train":
                    scheduler.step()

                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                epoch_acc = running_corrects.double() / len(
                    dataloaders[phase].dataset
                )

                print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

                # deep copy the model
                if phase == "val" and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(self.model.state_dict())

            print()

        time_elapsed = time.time() - since
        print(
            f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s"
        )
        print(f"Best val Acc: {best_acc:4f}")

        # load best model weights
        self.model.load_state_dict(best_model_wts)
        return self.model

    def accuracy(self, dataloader: torch.utils.data.DataLoader = None):
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

        with torch.no_grad():
            for batch in dataloader:
                images, labels = batch
                outputs = self.model(images.to(self.device))
                _, predicts = torch.max(outputs.data, 1)
                total += labels.size(0)
                corrects += (predicts.cpu() == labels).sum().item()

        return corrects / total

    def load(
        self,
        file: str = "weights.pkl",
        drive: bool = True,
        drive_id: str = "1SpWaLYVv6CqKT-9uTA2PoLLvG_CzeIU9",
    ) -> nn.Module:
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

        drive_id: (str)
        Id do drive que está hospedado
        drive = True.

        """
        if drive:
            gdown.download(DRIVE_URL + drive_id, file, quiet=False)

        model_st = torch.load(file, map_location=self.device)
        self.model.load_state_dict(model_st)

    def save(self, filename: str, path: str = "") -> Type[None]:
        """
        Salva um PyTorch model, no path e nome desejados, com extensão .pkl
        ---------------
        Argumentos:
            - model: PyTorch model (torch.nn.Module)
            - path: string contendo o path para salvar o modelo
            - filename: string contendo o nome do arquivo salvo
        """
        filepath = Path(path) / filename
        torch.save(self.model.state_dict(), filepath)

    def _tensor_loader(self, image: Union[str, Path, bytes]):
        """
        Descrição
        --------
        Carrega o tensor a partir do endereço
        da imagem e do transformador
        Entradas
        --------
        image: str | Path | bytes | PIL.Image
        Imagem a ser vonvertida
        Saídas
        ------
        image: torch.Tensor
        Tensor em torch da imagem carregada
        """
        if isinstance(image, bytes):
            image = Image.fromarray(image)
        elif isinstance(image, (str, Path)):
            print("Open image")
            image = Image.open(image)

        image = transforms.Compose(self.transforms)(image.convert("RGB"))
        image = Variable(image, requires_grad=False)
        image = image.unsqueeze(0)
        return image

    def predict(
        self,
        image: Union[str, Path, bytes],
        verbose: bool = False,
    ) -> (torch.Tensor, str):
        """
        Descrição
        --------
        Encontra a predição da rede para apenas
        uma imagem em análise
        Entradas
        --------
        image: Union[str, Path, bytes, Image]
        Imagem a ser categorizada

        verbose: bool
        Variável que ativa o print das imagens
        focando apenas no retorno
        Saídas
        ------
        outputs: torch.Tensor
        Tensor com o output da rede para a imagem
        passada
        lable: str
        Label da resposta
        """
        self.model.eval()
        with torch.no_grad():
            image = self._tensor_loader(image).to(self.device)
            outputs = self.model(image)
            _, preds = torch.max(outputs, 1)
            label = self.class_names[preds]
            if verbose:
                tensor_imshow(
                    image.cpu().data[0],
                    f"predicted: {label}",
                )

        return outputs, label


def tensor_imshow(tensor: torch.Tensor, title: str = None):
    """
    Descrição
    --------
    Mostra a imagem em tensores, ajustando
    seus valores de média e desvio padrão
    Entradas
    --------
    tensor: torch.Tensor
    Tensor em torch que deve ser mostrado
    title: str
    Título a ser mostrado pela imagem
    Saídas
    ------
    None
    """

    tensor = tensor.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    tensor = std * tensor + mean
    tensor = np.clip(tensor, 0, 1)
    plt.imshow(tensor)
    plt.axis("off")
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated
    plt.show()
