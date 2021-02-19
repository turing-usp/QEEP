"""
Site base: https://bulbapedia.bulbagarden.net/wiki/Bulbapedia
"""

from typing import List
import requests
from pokedex import pokedex
from hashlib import md5


def _imgName2url(imgName: str):
    """
    Descrição
    --------
    Gera a url para baixar a partir do nome da imagem conforme descrito na documentação da mediawiki

    Entradas
    --------
    imgName: str
    nome da imagem

    Saídas
    ------
    url: str
    URL da imagem para ser baixada

    """

    filename = imgName.strip("File:").replace(" ", "_").encode("utf-8")

    # a estrutura do diretório da imagem é composto usando o hash md5 do nome do arquivo.
    # Documentado em: https://www.mediawiki.org/wiki/Manual:$wgHashedUploadDirectory
    hashedFilename = md5(filename).hexdigest()
    link = f"https://archives.bulbagarden.net/media/upload/{hashedFilename[0]}/{hashedFilename[0:2]}/{filename.decode('utf-8')}"
    return link


def getImagesURLbyId(id: int) -> List[str]:
    """
    Descrição
    --------
    Descobre todas as imagens de um pokemon em https://bulbapedia.bulbagarden.net/wiki/Bulbapedia

    Entradas
    --------
    id: int
    Numero da pokedex do pokemon

    Saídas
    ------
    urls: List<str>
    Lista de urls encontradas

    """
    print(f"> Pushando #{id} de bulbagarden.com")

    pokemon = pokedex[id]

    url = "https://archives.bulbagarden.net/w/api.php"
    params = {
        "action": "query",
        "list": "categorymembers",
        "cmtitle": "Category:" + pokemon.name,
        "cmlimit": "max",  # padrão 500
        "cmtype": "file",
        "format": "json",
    }

    # continua o scraping caso haja mais imagens do que o cmlimit
    resp = {"continue": {"cmcontinue": ""}}
    links = []
    while "continue" in resp:
        params["cmcontinue"] = resp["continue"]["cmcontinue"]
        resp = requests.get(url, params).json()
        members = resp["query"]["categorymembers"]
        links += [_imgName2url(image["title"]) for image in members]
    return links


if __name__ == "__main__":
    for id in range(1, 2):
        urls = getImagesURLbyId(id)
        print(len(urls))
        print(*urls, sep="\n")
