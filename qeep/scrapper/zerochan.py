"""
Site base: https://www.zerochan.net/pikachu?p=2
"""

from typing import List
import requests
from bs4 import BeautifulSoup
from util.pokedex import pokedex


def get_images_url_by_id(pokemon_id: int) -> List[str]:
    """
    Descrição
    --------
    Descobre todas as imagens de um pokemon em https://zerochan.net

    Entradas
    --------
    pokemon_id: int
    Numero da pokedex do pokemon

    Saídas
    ------
    urls: List<str>
    Lista de urls encontradas

    """
    pokemon = pokedex[pokemon_id]

    url = f"https://www.zerochan.net/{pokemon.name}"

    page = 0
    links = []
    while True:
        page += 1
        resp = requests.get(f"{url}?p={page}", timeout=10)

        if resp.status_code != 200:
            break

        soup = BeautifulSoup(resp.text, features="lxml")
        imgs = soup.find_all("img", {"alt": pokemon.name.capitalize()})

        if len(imgs) == 0:
            break

        links += [img.get("src") for img in imgs]
    return links
