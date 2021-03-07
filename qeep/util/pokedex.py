"""
    Ao importar roda _fillPokedex(),
    prenchendo a variavel local pokedex com um dicionario de pokemons
"""

import requests

pokedex = None


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
    id: int = None
    name: str = None

    def __init__(self, id: str, name: str):
        self.id = id
        self.name = name

    def __str__(self):
        return f"#{self.id:03}-{self.name}"

    def __repr__(self):
        return f"Pokemon({self.id},{self.name})"


def _fillPokedex():
    """
        Prenche a variavel global com os pokemons oriundos de um csv
    """
    global pokedex
    if pokedex is not None:
        return

    pokedex = dict()

    response = requests.get(
        "https://raw.githubusercontent.com/veekun/pokedex/master/pokedex/data/csv/pokemon.csv")

    # Break in lines and remove header
    pokedexData = response.text.split('\n')[1:]
    for pokedexLine in pokedexData:
        if pokedexLine.strip(' ,') == '':
            continue

        [id, name, *_] = pokedexLine.split(',')
        id = int(id)  # Grr odeio a tipagem fraca de python
        pokemon = Pokemon(id, name)
        pokedex[id] = pokemon
        pokedex[name] = pokemon


_fillPokedex()
