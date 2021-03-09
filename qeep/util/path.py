from pathlib import Path
import os
from typing import Union


def createDirIfNotExist(path: Union[Path, str]) -> None:
    """
    Descrição
    --------
    Cria um diretorio se ele não existe

    Entradas
    --------
    path: Path | str
    Path do diretorio que será avaliado

    Saídas
    ------
    None

    """
    if path is str:
        path = Path(path)

    if not path.exists():
        print("> Create dir:", path)
        os.makedirs(path)
