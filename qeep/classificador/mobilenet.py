import torch
import torch.nn as nn
from ..util.model_validation import ModelValidation
from ..util.model_dataset import ModelDataset
from ..util.model_storage import ModelStorage


class MobileNet(ModelValidation, ModelStorage, ModelDataset):
    def __init__(self, output_size: int):
        model = torch.hub.load('pytorch/vision:v0.6.0',
                               'mobilenet_v2', pretrained=True)

        num_ftrs = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(num_ftrs, output_size)

        # model_output = model.classifier[-1].out_features
        # model.classifier.append(nn.Softmax(output_size))

        self.model = model.to(self.device)


if __name__ == "__main__":
    mobilenet = MobileNet(151)
    # mobilenet.loadModel("mobilenet_weights.pkl")
    mobilenet.show()
    mobilenet.dataset_load_all()
    mobilenet.trainModel(epochs=5)
    mobilenet.saveModel("mobilenet_weights.pkl")
