import json
from dataset import loadDataset


def saveClasses(classesPath: str = "../classes.json", datasetPath: str = "./data'"):
    """
    Descrição
    --------
    Salva em json as classes do dataset

    Entradas
    --------
    classesPath: str
    Arquivo que será salvo as classes

    datasetPath: str
    Diretorio base do dataset

    Saídas
    ------
    None

    """
    dataset = loadDataset(datasetPath, None)
    with open(classesPath, mode="w") as f:
        json.dump(dataset.classes, f)


def loadClasses(classesPath: str = "../classes.json"):
    """
    Descrição
    --------
    Carrega as classes a partir do json

    Entradas
    --------
    classesPath: str
    Local para o arquivo das classes

    Saídas
    ------
    List<str>
    Lista com as classes

    """
    with open(classesPath, mode="r") as f:
        classes = json.load(f)

    return classes


if __name__ == "__main__":
    saveClasses()
