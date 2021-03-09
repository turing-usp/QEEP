import torch
from torch.utils.data import DataLoader


def modelAccuracy(model: torch.nn.Module, model_dataloader: DataLoader,
                  device: str = "cpu"):
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
