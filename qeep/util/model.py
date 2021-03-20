from torchvision import transforms
from typing import Type, List
import copy
import gdown
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim


class ModelUtil:
    """
    Classe basica para manipular, treinar, e carregar modelos
    """

    model: torch.nn.Module

    def __call__(self, x):
        self.model(x)

    def show(self):
        print(self.model.eval())

    @property
    def device(self) -> str:
        return "cuda:0" if torch.cuda.is_available() else "cpu"

    @property
    def transforms(self) -> List[torch.nn.Module]:
        return [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]

    def train(
        self,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler.StepLR,
        train_dataloader: torch.utils.data.DataLoader,
        val_dataloader: torch.utils.data.DataLoader,
        epochs: int = 25,
    ):

        since = time.time()

        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0

        dataloaders = {"train": train_dataloader, "val": val_dataloader}

        for epoch in range(epochs):
            print("Epoch {}/{}".format(epoch, epochs - 1))
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

                epoch_loss = running_loss
                epoch_acc = running_corrects.double()

                print(
                    "{} Loss: {:.4f} Acc: {:.4f}".format(
                        phase, epoch_loss, epoch_acc
                    )
                )

                # deep copy the model
                if phase == "val" and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(self.model.state_dict())

            print()

        time_elapsed = time.time() - since
        print(
            "Training complete in {:.0f}m {:.0f}s".format(
                time_elapsed // 60, time_elapsed % 60
            )
        )
        print("Best val Acc: {:4f}".format(best_acc))

        # load best model weights
        self.model.load_state_dict(best_model_wts)
        return self.model

        print(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
        print(f"Validation Accuracy: {self.accuracy()}")
        print()

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
        url: str = "https://drive.google.com/uc?export=download&id=1yC0qK0gVX5sc6GTpBPBupH3TSFssLkqS",
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

    def save(self, filename: str, path: str = "") -> Type[None]:
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
