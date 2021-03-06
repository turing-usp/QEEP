"""
Site base: https://pokemon.com
"""

from typing import List
from repository import downloadImgs


def getImagesURLbyId(id: int) -> List[str]:
    """
    Descrição
    --------
    Descobre todas as imagens de um pokemon em https://pokemon.com

    Entradas
    --------
    id: int
    Numero da pokedex do pokemon

    Saídas
    ------
    urls: List<str>
    Lista de urls encontradas

    """

    print(f"> Pushando #{id} de pokemon.com")

    return [f"https://assets.pokemon.com/assets/cms2/img/pokedex/full/{id:03}.png"]


def getImagesbyId(id: int) -> List[bytes]:
    """
    Descrição
    --------
    Descobre todas as imagens de um pokemon em https://pokemon.com e baixa

    Entradas
    --------
    id: int
    Numero da pokedex do pokemon

    Saídas
    ------
    urls: List<str>
    Lista de urls encontradas

    """
    urls = getImagesURLbyId(id)
    imgs = downloadImgs(urls)
    return imgs


if __name__ == "__main__":
    for id in range(1, 3):
        print(getImagesURLbyId(id))
