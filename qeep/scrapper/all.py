"""
    Funções que unem as demais funções do modulo
"""

from typing import List
from pathlib import Path
from iteration_utilities import deepflatten

from qeep.scrapper.gameinfo import GameInfoScrapper
from qeep.scrapper.pokemondb import PokemonDBScrapper
from qeep.scrapper.pokemon_oficial import PokemonOficialScrapper
from qeep.scrapper.pokemon_oficial_cards import PokemonOficialCardsScrapper
from qeep.scrapper.serebii import SerebiiScrapper

from qeep.pokedex import pokedex  # noqa: PLE0611, PLE0401
from qeep.scrapper.scrapper import Scrapper, resilient_session, Session

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
        "Retorna todas as urls de pokemon"
        return deepflatten(
            [s.get_images_url() for s in self.scrappers], depth=1
        )

    def get_images(self) -> List[bytes]:
        "Retorna todas as imagens de pokemons"
        return deepflatten([s.get_images() for s in self.scrappers], depth=1)


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
        poke_scrapper = PokemonScrapper(pokemon_id, req_session)
        poke_path.mkdir(parents=True)
        return poke_scrapper.save(poke_path)

    base_path.mkdir(parents=True)
    for i in ids:
        save_pokemon_images(i)
