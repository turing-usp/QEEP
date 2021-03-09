import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import ..util.train as train
from ..util.storage import saveModel, loadModel
from .dataset import loadSplitedLoader

class MobileNet():
    def __init__(self):
        pass

    def loadDataset(self, *args, **kwargs):
        *self.data_loaders, self.dataset = loadSplitedLoader(*args, **kwargs)
        self.classes_len = len(dataset.classes)

    def trainModel(self, batch_size=4, criterion=None, optimizer=None, scheduler=None,
                   epochs=25, learning_rate=0.001):
        if criterion is None:
            criterion = nn.CrossEntropyLoss()
        if optimizer is None:
            optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9)
        if scheduler is None:
            scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

        train.trainModel(self.model, self.data_loaders, criterion, optimizer, scheduler, epochs)

    def createMobilenet(data_loaders=None, dataset=None):
        instance = MobileNet()
        if dataset is None:
            instance.LoadDataset()
        else:
            instance.data_loaders = data_loaders
            instance.dataset = dataset
            instance.classes_len = len(dataset.classes)

        instance.model = torch.hub.load('pytorch/vision:v0.6.0',
                                        'mobilenet_v2', pretrained=True)

        num_ftrs = instance.model.fc.in_features

        instance.model.fc = nn.Linear(num_ftrs, instance.classes_len)

        return instance


    def loadPretrained(model="mobilenet", file: str = "mobilenet_weights.pkl", drive: bool = True,
                       url: str ="https://drive.google.com/uc?export=download&id=1yC0qK0gVX5sc6GTpBPBupH3TSFssLkqS")
        instance = MobileNet()
        instance.model = loadModel(model, file, drive, url)
        return instance
