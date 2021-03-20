"""
Site base: https://www.pokemon.com/us/pokemon-tcg/pokemon-cards/?cardName=bulbasaur
"""

from typing import List
import requests
import re
from bs4 import BeautifulSoup
from ..util.pokedex import pokedex
from ..util.image_repository import downloadImgs
from PIL import Image
import io

filterCards = "basic-pokemon=on&stage-1-pokemon=on&stage-2-pokemon=on&level-up-pokemon=on&ex-pokemon=on&mega-ex=on&special-pokemon=on&pokemon-legend=on&restored-pokemon=on&break=on&pokemon-gx=on&pokemon-v=on&pokemon-vmax=on"


def getImagesURLbyId(id: int) -> List[str]:
    """
    Descrição
    --------
    Descobre todas as imagens de um pokemon em https://www.pokemon.com/us/pokemon-tcg/pokemon-cards/

    Entradas
    --------
    id: int
    Numero da pokedex do pokemon

    Saídas
    ------
    urls: List<str>
    Lista de urls encontradas

    """

    print(f"> Pushando #{id} de pokemon.com/us/pokemon-tcg/pokemon-cards")
    pokemon = pokedex[id]

    def url(page=1):
        return f"https://www.pokemon.com/us/pokemon-tcg/pokemon-cards/{page}?cardName={pokemon.name}&{filterCards}"

    page = 0
    links = []
    while True:
        page += 1
        response = requests.get(url(page))

        if response.status_code != 200:
            break

        soup = BeautifulSoup(response.text, features="lxml")
        imgs = soup.find_all(
            "img",
            {
                "src": re.compile(
                    "https://assets.pokemon.com/assets/cms2/img/cards/web/"
                )
            },
        )
        local_links = [img.get("src") for img in imgs]

        # A ultima pagina é repetida quando vc acessa uma pagina que não existe
        if len(local_links) == 0 or local_links[-1] in links:
            break

        links += local_links

    return links


def crop(imgBytes: bytes) -> bytes:
    """
    Descrição
    --------
    Corta apenas o quadrado do pokemon

    Entradas
    --------
    img: bytes
    Imagem em bytes

    Saídas
    ------
    img: List<bytes>
    Image cortada

    """
    img = Image.open(io.BytesIO(imgBytes))
    img = img.crop((20, 35, 225, 175))

    # https://stackoverflow.com/questions/33101935/convert-pil-image-to-byte-array#33117447
    imgByteOut = io.BytesIO()
    img.save(imgByteOut, format="PNG")
    return imgByteOut.getvalue()


def getImagesbyId(id: int) -> List[bytes]:
    """
    Descrição
    --------
    Descobre todas as imagens de um pokemon em https://www.pokemon.com/us/pokemon-tcg/pokemon-cards/ e baixa

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
    croppedImage = [crop(img) for img in imgs]
    return croppedImage


if __name__ == "__main__":
    for id in range(3, 4):
        urls = getImagesURLbyId(id)
        print(len(urls))
        print(*urls, sep="\n")
