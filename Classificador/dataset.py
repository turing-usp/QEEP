import torch
from torchvision import transforms, datasets
from pathlib import Path

torch.manual_seed(4)

default_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def loadDataset(path: Path, tranform: torch.nn.Module):
    """
    Descrição
    --------
    Carrega o Dataset

    Entradas
    --------
    path: Path
    Diretorio outro diretorio dentro, cada um uma classe

    tranform: torch Tranform
    Transformaçoes a serem aplicadas no dataset
    Se não for defenido será usado o default_ttranform

    Saídas
    ------
    torch Dataset
    Dataset carregado

    """
    if not path.exists():
        raise Exception('Dataset not found')

    if tranform is None:
        tranform = default_transform

    return datasets.ImageFolder(root=path,
                                transform=tranform)


def loadSplitedDataset(tresh_hold: float, path: Path, transform: torch.nn.Module):
    """
    Descrição
    --------
    Carrega o Dataset e separa ele em dois grupos: de treino e validação

    Entradas
    --------
    tresh_hold: float
    Porcentagem de treino em relação ao dataset original

    path: Path
    Diretorio outro diretorio dentro, cada um uma classe

    tranform: torch Tranform
    Transformaçoes a serem aplicadas no dataset

    Saídas
    ------
    torch Dataset
    Dataset de treino

    torch Dataset
    Dataset de validação
    """

    dataset = loadDataset(path, transform)
    split_dataset = torch.utils.data.random_split(
        dataset, [int(len(dataset) * tresh_hold), int(len(dataset) * (1-tresh_hold))])
    return split_dataset[0], split_dataset[1]


def loadSplitedLoader(batch_size: int, num_workers: int, tresh_hold: float, path: Path, transform: torch.nn.Module):
    """
    Descrição
    --------
    Carrega o Dataset e separa ele em dois grupos: de treino e validação e os transforma em bachs

    Entradas
    --------
    batch_size: int
    Tamanho de cada batch

    num_workers: int
    Quantidade de workres

    tresh_hold: float
    Porcentagem de treino em relação ao dataset original

    path: Path
    Diretorio outro diretorio dentro, cada um uma classe

    tranform: torch Tranform
    Transformaçoes a serem aplicadas no dataset

    Saídas
    ------
    torch Dataset
    Dataset de treino

    torch Dataset
    Dataset de validação
    """

    train_dataset, val_dataset, datasets = loadSplitedDataset(
        tresh_hold, path, transform)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return train_loader, val_loader, datasets
