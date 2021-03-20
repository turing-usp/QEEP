"""
    PokemonDBScrapper
"""

from typing import List
import re
from bs4 import BeautifulSoup
from ..util.pokedex import pokedex
from .scrapper import Scrapper, Session


class PokemonDBScrapper(Scrapper):
    """Site base: https://pokemondb.net"""

    def __init__(self, pokemon_id: int, session: Session):
        Scrapper.__init__(self, pokemon_id, session)

    def get_images_url(self) -> List[str]:
        """ Descobre todas as imagens no site"""

        pokemon = pokedex[self.pokemon_id]
        url = f"https://pokemondb.net/sprites/{pokemon.name}"

        response = self.session.get(url, timeout=10)
        soup = BeautifulSoup(response.text, features="lxml")

        pattern = r"https://img.pokemondb.net/sprites/.*"
        imgs = soup.find_all("img", {"src": re.compile(pattern)})
        lazy_imgs = soup.find_all("span", {"data-src": re.compile(pattern)})

        links = []
        links += [img.get("src") for img in imgs]
        links += [img.get("data-src") for img in lazy_imgs]

        return links
