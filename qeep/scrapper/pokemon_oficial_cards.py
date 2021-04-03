"""
Site base: https://www.pokemon.com/us/pokemon-tcg/pokemon-cards/?cardName=bulbasaur
"""

import re
import io
from typing import List
from bs4 import BeautifulSoup
from PIL import Image
from qeep.scrapper.scrapper import Scrapper, Session
from qeep.pokedex import pokedex  # noqa: PLE0611, PLE0401

FILTER_CARDS = "basic-pokemon=on&stage-1-pokemon=on&stage-2-pokemon=on&level-up-pokemon=on&ex-pokemon=on&mega-ex=on&special-pokemon=on&pokemon-legend=on&restored-pokemon=on&break=on&pokemon-gx=on&pokemon-v=on&pokemon-vmax=on"


class PokemonOficialCardsScrapper(Scrapper):
    """Site base: https://pokemon.com"""

    def __init__(self, pokemon_id: int, session: Session):
        Scrapper.__init__(self, pokemon_id, session)

    def get_images_url(self) -> List[str]:
        """ Descobre todas as imagens no site"""
        pokemon = pokedex[self.pokemon_id]

        def url(page=1):
            return f"https://www.pokemon.com/us/pokemon-tcg/pokemon-cards/{page}?cardName={pokemon.name}&{FILTER_CARDS}"

        page = 0
        urls = []
        while True:
            page += 1
            response = self.session.get(url(page), timeout=10)

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
            local_urls = [img.get("src") for img in imgs]

            # A ultima pagina é repetida quando vc acessa uma pagina que não existe
            if len(local_urls) == 0 or local_urls[-1] in urls:
                break

            urls += local_urls

        return urls

    @staticmethod
    def _img_processing(img: bytes) -> bytes:
        """Corta apenas o quadrado do pokemon"""
        img_pil = Image.open(io.BytesIO(img))
        img_pil = img_pil.crop((20, 35, 225, 175))

        # https://stackoverflow.com/questions/33101935/convert-pil-image-to-byte-array#33117447
        img_byte_out = io.BytesIO()
        img_pil.save(img_byte_out, format="PNG")
        return img_byte_out.getvalue()
