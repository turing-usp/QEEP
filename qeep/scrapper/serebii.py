"""
Site base: https://www.serebii.net
"""

from typing import List
import requests
import re
from bs4 import BeautifulSoup
from ..util.image_repository import downloadImgs


def getImagesURLbyId(id: int) -> List[str]:
    """
    Descrição
    --------
    Descobre todas as imagens de um pokemon em https://serebii.net

    Entradas
    --------
    id: int
    Numero da pokedex do pokemon

    Saídas
    ------
    urls: List<str>
    Lista de urls encontradas

    """
    print(f"> Pushando #{id} de serebii.net")

    url = f"https://www.serebii.net/card/dex/{id:03}.shtml"

    response = requests.get(url, timeout=10)
    soup = BeautifulSoup(response.text, features="lxml")
    imgs = soup.find_all("img", {"src": re.compile("/card/th/.*.jpg")})
    links = ["https://www.serebii.net" + img.get("src") for img in imgs]

    artURL = f"https://www.serebii.net/art/th/{id}.png"
    return [artURL] + links


def getImagesbyId(id: int) -> List[bytes]:
    """
    Descrição
    --------
    Descobre todas as imagens de um pokemon em https://serebii.net e baixa

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
