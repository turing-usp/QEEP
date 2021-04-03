"""
    Ao importar roda _fillPokedex(),
    prenchendo a variavel local pokedex com um dicionario de pokemons
"""

from typing import Dict, Union
import requests


class Pokemon:
    """
    Descrição
    --------
    Classe basica para agrupar dados de pokemons

    Atributos
    --------
    id: int
    Numero da pokedex do pokemon

    name: str
    Nome do pokemon
    """

    pokeid: int = None
    name: str = None

    def __init__(self, pokemon_id: str, name: str):
        self.pokeid = pokemon_id
        self.name = name

    def __str__(self):
        return f"#{self.pokeid:03}-{self.name}"

    def __repr__(self):
        return f"Pokemon({self.pokeid},{self.name})"


def _fill_pokedex(acc):
    """
    Prenche a variavel global com os pokemons oriundos de um csv
    """
    if len(acc) > 0:
        return

    response = requests.get(
        "https://raw.githubusercontent.com/veekun/pokedex/master/pokedex/data/csv/pokemon.csv",
        timeout=10,
    )

    # Break in lines and remove header
    pokedex_data = response.text.split("\n")[1:]
    for pokedex_line in pokedex_data:
        if pokedex_line.strip(" ,") == "":
            continue

        [pokeid, name, *_] = pokedex_line.split(",")
        pokeid = int(pokeid)  # Grr odeio a tipagem fraca de python
        pokemon = Pokemon(pokeid, name)
        acc[pokeid] = pokemon
        acc[name] = pokemon


pokedex: Dict[Union[str, int], Pokemon] = {}
_fill_pokedex(pokedex)
