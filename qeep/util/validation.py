import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.autograd import Variable
from PIL import Image
from typing import List


def modelAccuracy(
    model: torch.nn.Module, model_dataloader: DataLoader, device: str = "cpu"
):
    """
    Descrição
    --------
    Função que calcula a acurácia de um modelo dado um dataloader e o device.

    Entradas
    --------
    model: (torch.nn.Module)
    O modelo que está sendo avaliado.

    model_dataloader: (torch.Tensor)
    Dataloader que contém o conjunto de imagens e rótulos.

    device: (String)
    String que representa o dispotivo acelerador de hardware.

    Saídas
    ------
    accuracy: (float)
    Acurácia do modelo para o conjunto de dados do dataloader.
    """
    corrects = 0
    total = 0

    with torch.no_grad():
        for batch in model_dataloader:
            images, labels = batch
            outputs = model(images.to(device))
            _, predicts = torch.max(outputs.data, 1)
            total += labels.size(0)
            corrects += (predicts.cpu() == labels).sum().item()

    return 100 * (corrects / total)


def _testPokeLoader() -> transforms.Compose:
    """
    Descrição
    --------
    Gera o objeto compose para transformar a
    imagem em tensor
    Entradas
    --------
    Saídas
    ------
    loader: transforms.Compose
    """

    loader = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    return loader


def tensorImshow(tensor: torch.Tensor, title: str = None):
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


def _tensorLoader(image_array: List[int], loader: transforms.Compose):
    """
    Descrição
    --------
    Carrega o tensor a partir do endereço
    da imagem e do transformador
    Entradas
    --------
    image_name: str
    Caminho até a imagem a ser convertida
    em tensor
    loader: transforms.Compose
    Torch compose para normalizar a imagem,
    transformar a mesma em tensor e dar
    resize
    Saídas
    ------
    image: torch.Tensor
    Tensor em torch da imagem carregada
    """
    image = Image.fromarray(image_array, mode="RGB")
    image = loader(image)
    image = Variable(image, requires_grad=False)
    image = image.unsqueeze(0)
    return image


def _getOnePrediction(
    model: torch.nn,
    class_names: List[str],
    image_array: List[int],
    loader: transforms.Compose = None,
    quiet: bool = False,
) -> torch.Tensor:
    """
    Descrição
    --------
    Encontra a predição da rede para apenas
    uma imagem em análise
    Entradas
    --------
    model: torch.nn
    Objeto da rede a ser utilizada para
    obter a predição
    class_names: List[str]
    Vetor que contêm as classes do dataset
    para ser plotado
    loader: transforms.Compose
    Torch compose para normalizar a imagem,
    transformar a mesma em tensor e dar
    resize
    image_name: str
    Caminho até a imagem a ser carregada
    quiet: bool
    Variável que desativa o print das imagens
    focando apenas no retorno
    Saídas
    ------
    outputs: torch.Tensor
    Tensor com o output da rede para a imagem
    passada
    """
    # model.eval()
    with torch.no_grad():
        image = _tensorLoader(image_array, loader)
        outputs = model(image)
        _, preds = torch.max(outputs, 1)
        if not quiet:
            tensorImshow(
                image.cpu().data[0], "predicted: {}".format(class_names[preds])
            )

    return outputs


def getPredictions(
    model: torch.nn,
    class_names: List[str],
    image_arrays: List[int],
    quiet: bool = False,
) -> List[torch.Tensor]:
    """
    Descrição
    --------
    Encontra a predição da rede para uma ou várias
    imagens em análise
    Entradas
    --------
    model: torch.nn
    Objeto da rede a ser utilizada para
    obter a predição
    class_names: List[str]
    Vetor que contêm as classes do dataset
    para ser plotado
    image_name: List[str]
    Vetor de caminhos até as imagens a
    serem carregadas
    quiet: bool
    Variável que desativa o print das imagens
    focando apenas no retorno
    Saídas
    ------
    outputs: List[torch.Tensor]
    Lista de tensores com os outputs da rede
    para as imagens passadas
    """

    outputs = []
    loader = _testPokeLoader()

    for image_array in image_arrays:
        outputs.append(
            _getOnePrediction(
                model,
                class_names=class_names,
                loader=loader,
                image_array=image_array,
                quiet=quiet,
            )
        )
    return outputs


if __name__ == "__main__":
    _testPokeLoader()
    image = Image.open("pikachu.png")
    image = np.array(image, dtype="int8")
    image = Image.fromarray(image, mode="RGB")
    print(image)

    image = _tensorLoader(image, _testPokeLoader())
    print(image)
