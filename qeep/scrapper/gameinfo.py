"""
Site base: https://gameinfo.io
"""

from typing import List
from ..util.image_repository import downloadImgs


def getImagesURLbyId(id: int) -> List[str]:
    """
    descrição
    --------
    descobre todas as imagens de um pokemon em https://gameinfo.io

    entradas
    --------
    id: int
    numero da pokedex do pokemon

    saídas
    ------
    urls: list<str>
    lista de urls encontradas

    """
    print(f"> Pushando #{id} de gamainfo.io")

    normalImg = f"https://images.gameinfo.io/pokemon/256/{id:03}-00.png"
    shineImg = f"https://images.gameinfo.io/pokemon/256/{id:03}-00-shiny.png"
    return [normalImg, shineImg]


def getImagesbyId(id: int) -> List[bytes]:
    """
    descrição
    --------
    descobre todas as imagens de um pokemon em https://gameinfo.io e as baixa

    entradas
    --------
    id: int
    numero da pokedex do pokemon

    saídas
    ------
    imgs: list<bytes>
    lista de imagens

    """
    urls = getImagesURLbyId(id)
    imgs = downloadImgs(urls)
    return imgs


if __name__ == "__main__":
    for id in range(1, 3):
        print(getImagesURLbyId(id))
