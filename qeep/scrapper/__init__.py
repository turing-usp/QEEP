"""
    Funções que unem as demais funções do modulo
"""

from typing import List
from pathlib import Path
from multiprocessing import Pool
from iteration_utilities import deepflatten

from .gameinfo import GameInfoScrapper
from .pokemondb import PokemonDBScrapper
from .pokemon_oficial import PokemonOficialScrapper
from .pokemon_oficial_cards import PokemonOficialCardsScrapper
from .serebii import SerebiiScrapper

from ..util.pokedex import pokedex
from ..util.path import create_dir_if_not_exist
from .scrapper import Scrapper, resilient_session, Session

_NUMBER_POOLS = 6


class PokemonScrapper(Scrapper):
    """União de varios scrappers"""

    scrappers: List[Scrapper]

    def __init__(self, pokemon_id: int, session: Session):
        Scrapper.__init__(self, pokemon_id, session)
        self.scrappers = [
            PokemonOficialScrapper(pokemon_id, session),
            PokemonOficialCardsScrapper(pokemon_id, session),
            GameInfoScrapper(pokemon_id, session),
            PokemonDBScrapper(pokemon_id, session),
            SerebiiScrapper(pokemon_id, session),
        ]

    def get_images_url(self) -> List[str]:
        return deepflatten([s.get_images_url() for s in self.scrappers])

    def get_images(self) -> List[bytes]:
        return deepflatten([s.get_images() for s in self.scrappers])


req_session = resilient_session()


def get_all_images_and_save_by_ids(ids: List[int], base_path: Path):
    """
    Descrição
    --------
    Descobre todas as imagens de uma faixa pokemon nos scrappers criados e salva elas
    de maneira paralela a cada pokemon

    Entradas
    --------
    ids: List<int>
    Lista de Numero da pokedex do pokemon

    base_path: Path
    Diretorio em que será criado uma pasta com o nome do pokemon e salvara as imagens
    (Cria se o diretorio não existir)
    """

    def save_pokemon_images(pokemon_id: int):
        poke_path = base_path / pokedex[pokemon_id].name
        create_dir_if_not_exist(poke_path)
        PokemonScrapper(pokemon_id, req_session).save(poke_path)

    with Pool(_NUMBER_POOLS) as pool:
        pool.map(save_pokemon_images, ids)
