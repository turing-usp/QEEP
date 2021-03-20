"""
    GameInfoScrapper
"""

from typing import List
from .scrapper import Scrapper, Session


class GameInfoScrapper(Scrapper):
    """Site base: https://gameinfo.io"""

    def __init__(self, pokemon_id: int, session: Session):
        Scrapper.__init__(self, pokemon_id, session)

    def get_images_url(self) -> List[str]:
        """ Descobre todas as imagens no site"""

        normal_img = f"https://images.gameinfo.io/pokemon/256/{self.pokemon_id:03}-00.png"
        shine_img = f"https://images.gameinfo.io/pokemon/256/{self.pokemon_id:03}-00-shiny.png"
        return [normal_img, shine_img]
