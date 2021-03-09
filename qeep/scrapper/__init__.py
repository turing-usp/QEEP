from typing import List
from pathlib import Path
from functools import partial
from multiprocessing import Pool

# from . import bulbapedia
from . import gameinfo
from . import pokemondb
from . import pokemon
from . import pokeCards
from . import serebii
# from . import zerochan

from ..util import image_repository as repository
from ..util.pokedex import pokedex
from ..util.path import createDirIfNotExist

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


def getAllImagesbyId(id: int) -> List[bytes]:
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
    acc += pokemon.getImagesbyId(id)
    acc += gameinfo.getImagesbyId(id)
    acc += serebii.getImagesbyId(id)
    acc += pokeCards.getImagesbyId(id)
    acc += pokemondb.getImagesbyId(id)
    # acc += zerochan.getImagesbyId(id) # gera alguns lixos
    # acc += bulbapedia.getImagesbyId(id) # gera varios lixos
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
    imgs = getAllImagesbyId(id)
    path = base_path / pokemon.name
    createDirIfNotExist(path)
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
