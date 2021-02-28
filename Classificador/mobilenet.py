import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from Utils.storage import saveModel
from Utils.train import trainModel
from dataset import loadSplitedLoader


def createMobilenet(output_size, input_size, device):

    model = torch.hub.load('pytorch/vision:v0.6.0',
                           'mobilenet_v2', pretrained=True)
    model.eval()

    num_ftrs = model.fc.in_features

    model.fc = nn.Linear(num_ftrs, output_size)

    model = model.to(device)

    return model


if __name__ == "__main__":
    *data_loaders, dataset = loadSplitedLoader()
    classes_len = len(dataset.classes)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = createMobilenet(classes_len, 256*256, device)

    criterion = nn.CrossEntropyLoss()

    optimizer_ft = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    exp_lr_scheduler = lr_scheduler.StepLR(
        optimizer_ft, step_size=7, gamma=0.1)

    model = trainModel(model, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=25)

    saveModel("mobilenet", model, path="../models")
