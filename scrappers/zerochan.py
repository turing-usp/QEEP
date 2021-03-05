"""
Site base: https://www.zerochan.net/pikachu?p=2
"""

from typing import List
import requests
from pokedex import pokedex
from bs4 import BeautifulSoup
from repository import downloadImgs


def getImagesURLbyId(id: int) -> List[str]:
    """
    Descrição
    --------
    Descobre todas as imagens de um pokemon em https://zerochan.net

    Entradas
    --------
    id: int
    Numero da pokedex do pokemon

    Saídas
    ------
    urls: List<str>
    Lista de urls encontradas

    """

    print(f"> Pushando #{id} de zerochan.com")

    pokemon = pokedex[id]

    url = f"https://www.zerochan.net/{pokemon.name}"

    page = 0
    links = []
    while True:
        page += 1
        resp = requests.get(f"{url}?p={page}")

        if resp.status_code != 200:
            break

        soup = BeautifulSoup(resp.text, features="lxml")
        imgs = soup.find_all(
            "img", {"alt": pokemon.name.capitalize()})

        if len(imgs) == 0:
            break

        links += [img.get('src') for img in imgs]
    return links


def getImagesbyId(id: int) -> List[bytes]:
    """
    Descrição
    --------
    Descobre todas as imagens de um pokemon em https://zerochan.net e baixa

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
    for id in range(1, 4):
        urls = getImagesURLbyId(id)
        print(len(urls))
        print(*urls, sep="\n")
