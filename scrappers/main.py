#! /usr/bin/python

from typing import List
from pathlib import Path
from functools import partial
from multiprocessing import Pool

# import bulbapedia
import gameinfo
import pokemondb
import pokemon
import pokeCards
import serebii
# import zerochan

import repository
from pokedex import pokedex

_NUMBER_POOLS = 6


def getAllImagesURLbyId(id: int) -> List[str]:
    """
    Descrição
    --------
    Descobre todas as imagens de um pokemon nos scrappers criados

    Entradas
    --------
    id: int
    Numero da pokedex do pokemon

    Saídas
    ------
    urls: List<str>
    Lista de urls encontradas

    """

    print(f"> Pushando #{id}")
    acc = []
    acc += pokemon.getImagesURLbyId(id)
    acc += gameinfo.getImagesURLbyId(id)
    acc += serebii.getImagesURLbyId(id)
    acc += pokeCards.getImagesURLbyId(id)
    acc += pokemondb.getImagesURLbyId(id)
    # acc += zerochan.getImagesURLbyId(id) # gera alguns lixos
    # acc += bulbapedia.getImagesURLbyId(id) # gera varios lixos
    return acc


def getAllImagesAndSaveById(id: int, base_path: Path) -> List[Path]:
    """
    Descrição
    --------
    Descobre todas as imagens de um pokemon nos scrappers criados e salva elas

    Entradas
    --------
    id: int
    Numero da pokedex do pokemon

    base_path: Path
    Diretorio em que será criado uma pasta com o nome do pokemon e salvara as imagens
    (Cria se o diretorio não existir)

    Saídas
    ------
    urls: List<str>
    Lista de urls encontradas

    """
    pokemon = pokedex[id]
    imgsURL = getAllImagesURLbyId(id)
    imgs = repository.downloadImgs(imgsURL)
    path = base_path / pokemon.name
    repository.createDirIfNotExist(path)
    repository.writeImages(path, imgs)


def getAllImagesAndSaveByIds(ids: List[int], base_path: Path) -> List[Path]:
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

    Saídas
    ------
    urls: List<str>
    Lista de urls encontradas

    """
    f = partial(getAllImagesAndSaveById, base_path=base_path)
    with Pool(_NUMBER_POOLS) as p:
        p.map(f, ids)


def _main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Baixa imagens de pokemons em um diretorio')
    parser.add_argument('-b', '--begin', default=1,
                        type=int, help='pokemon inicial')
    parser.add_argument('-e', '--end', default=151, type=int,
                        help='ultimo pokemon a ser procurado')
    parser.add_argument('-p', '--path', default='../data',
                        type=str, help='diretorio que será salvo')

    args = parser.parse_args()
    path = Path(args.path)
    getAllImagesAndSaveByIds(range(args.begin, args.end+1), path)


if __name__ == "__main__":
    _main()
