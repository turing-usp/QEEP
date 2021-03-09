"""
Site base: https://pokemondb.net
"""

from typing import List
import requests
import re
from bs4 import BeautifulSoup
from ..util.pokedex import pokedex
from ..util.image_repository import downloadImgs


def getImagesURLbyId(id: int) -> List[str]:
    """
    Descrição
    --------
    Descobre todas as imagens de um pokemon em https://pokemondb.net

    Entradas
    --------
    id: int
    Numero da pokedex do pokemon

    Saídas
    ------
    urls: List<str>
    Lista de urls encontradas

    """

    print(f"> Pushando #{id} de pokemondb.net")

    pokemon = pokedex[id]
    url = f"https://pokemondb.net/sprites/{pokemon.name}"

    response = requests.get(url)
    soup = BeautifulSoup(response.text, features="lxml")

    pattern = r"https://img.pokemondb.net/sprites/.*"
    imgs = soup.find_all("img", {"src": re.compile(pattern)})
    lazy_imgs = soup.find_all("span", {"data-src": re.compile(pattern)})

    links = []
    links += [img.get('src') for img in imgs]
    links += [img.get('data-src') for img in lazy_imgs]

    return links


def getImagesbyId(id: int) -> List[bytes]:
    """
    Descrição
    --------
    Descobre todas as imagens de um pokemon em https://pokemondb.net e baixa

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
