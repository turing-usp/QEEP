import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.autograd import Variable
from PIL import Image
from typing import List

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

    loader = transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
    
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
    plt.axis('off')
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated
    plt.show()

def _tensorLoader(image_name: str, loader: transforms.Compose):
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
    
    image = Image.open(image_name)
    image = loader(image)
    image = Variable(image, requires_grad=False)
    image = image.unsqueeze(0)
    return image 
    
def _getOnePrediction(model: torch.nn, class_names: List[str], loader: transforms.Compose = None, image_name: str = "", quiet: bool = False) -> torch.Tensor:
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
    #model.eval()
    with torch.no_grad():
        image = _tensorLoader(image_name, loader)
        outputs = model(image)
        _, preds = torch.max(outputs, 1)
        if not quiet:
            tensorImshow(image.cpu().data[0], 'predicted: {}'.format(class_names[preds]))

    return outputs

def getPredictions(model: torch.nn, class_names: List[str], image_names: List[str], quiet: bool = False) -> List[torch.Tensor]:
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
    
    for image_name in image_names:
        outputs.append(_getOnePrediction(model, class_names = class_names, loader = loader, image_name = image_name, quiet = quiet))
    return outputs
