import torch
import torch.nn as nn
from ..util.model import ModelUtil


class MobileNet(ModelUtil):
    def __init__(self, output_size: int):
        model = torch.hub.load('pytorch/vision:v0.6.0',
                               'mobilenet_v2', pretrained=True)

        num_ftrs = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(num_ftrs, output_size)

        # model_output = model.classifier[-1].out_features
        # model.classifier.append(nn.Softmax(output_size))

        self.model = model.to(self.device)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Treina na mobilet')
    parser.add_argument('-t', '--train', default=True,
                        type=bool, help='Treino')
    parser.add_argument('-e', '--epochs', default=25,
                        type=int, help='Numero de epocas treinadas')
    parser.add_argument('-d', '--dataset', default="./data",
                        type=str, help='Path do dataset')
    parser.add_argument('-l', '--learningrate', default=0.001, type=float,
                        help='Learning Rate')
    parser.add_argument('-b', '--bathsize', default=4, type=int,
                        help='Bath Size')

    args = parser.parse_args()

    mobilenet = MobileNet(151)
    # mobilenet.loadModel("mobilenet_weights.pkl")
    mobilenet.show()
    if (args.train):
        mobilenet.dataset_load_all(path=args.dataset, batch_size=args.bathsize)
        mobilenet.trainModel(epochs=args.epochs, learning_rate=args.learningrate)
        mobilenet.saveModel("mobilenet_weights.pkl")
    else:
        # Parte que roda em uma imagem
        pass
