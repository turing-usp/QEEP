"""
    PokemonOficialScrapper
"""

from typing import List
from .scrapper import Scrapper, Session


class PokemonOficialScrapper(Scrapper):
    """Site base: https://pokemon.com"""

    def __init__(self, pokemon_id: int, session: Session):
        Scrapper.__init__(self, pokemon_id, session)

    def get_images_url(self) -> List[str]:
        """ Descobre todas as imagens no site"""
        return [
            f"https://assets.pokemon.com/assets/cms2/img/pokedex/full/{self.pokemon_id:03}.png"
        ]
