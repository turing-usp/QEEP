from typing import List
import re
from bs4 import BeautifulSoup
from qeep.scrapper.scrapper import Scrapper, Session


class SerebiiScrapper(Scrapper):
    """Site base: https://www.serebii.net"""

    def __init__(self, pokemon_id: int, session: Session):
        Scrapper.__init__(self, pokemon_id, session)

    def get_images_url(self) -> List[str]:
        """ Descobre todas as imagens no site"""
        url = f"https://www.serebii.net/card/dex/{self.pokemon_id:03}.shtml"

        response = self.session.get(url, timeout=10)
        soup = BeautifulSoup(response.text, features="lxml")
        imgs = soup.find_all("img", {"src": re.compile("/card/th/.*.jpg")})
        links = ["https://www.serebii.net" + img.get("src") for img in imgs]

        art_url = f"https://www.serebii.net/art/th/{self.pokemon_id}.png"
        return [art_url] + links
