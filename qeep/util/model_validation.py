from torch.optim import lr_scheduler
import gdown
import torch
import torch.nn as nn
import os
from typing import Type, List
from pathlib import Path
from torchvision import transforms, datasets
import torch.optim as optim
import copy
import time

default_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


class ModelValidation():
    """
    Classe basica para manipular modelos
    """
    name: str
    device: str
    model: nn.Module
    model: torch.nn.Module
    dataloaders: List[torch.utils.data.DataLoader]

    def __init__(self, name: str = "model"):
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

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
