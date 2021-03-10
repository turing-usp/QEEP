import torch
import torch.nn as nn
from ..util.model_validation import ModelValidation
from ..util.model_dataset import ModelDataset
from ..util.model_storage import ModelStorage


class MobileNet(ModelValidation, ModelStorage, ModelDataset):
    def __init__(self, output_size: int):
        self.model = torch.hub.load('pytorch/vision:v0.6.0',
                                    'mobilenet_v2', pretrained=True)

        num_ftrs = self.model.fc.in_features

        self.model.fc = nn.Linear(num_ftrs, output_size)
