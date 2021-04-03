"""
Site base: https://bulbapedia.bulbagarden.net/wiki/Bulbapedia
"""

from typing import List
from hashlib import md5
import requests
from qeep.pokedex import pokedex  # noqa: PLE0611, PLE0401


def _img_name_to_url(img_name: str):
    """
    Descrição
    --------
    Gera a url para baixar a partir do nome da imagem conforme descrito na documentação da mediawiki

    Entradas
    --------
    img_name: str
    nome da imagem

    Saídas
    ------
    url: str
    URL da imagem para ser baixada

    """

    filename = img_name.strip("File:").replace(" ", "_").encode("utf-8")

    # a estrutura do diretório da imagem é composto usando o hash md5 do nome do arquivo.
    # Documentado em: https://www.mediawiki.org/wiki/Manual:$wgHashedUploadDirectory
    hashed_filename = md5(filename).hexdigest()
    link = f"https://archives.bulbagarden.net/media/upload/{hashed_filename[0]}/{hashed_filename[0:2]}/{filename.decode('utf-8')}"

    return link


def get_image_url_by_id(pokemon_id: int) -> List[str]:
    """
    Descrição
    --------
    Descobre todas as imagens de um pokemon em https://bulbapedia.bulbagarden.net/wiki/Bulbapedia

    Entradas
    --------
    pokemon_id: int
    Numero da pokedex do pokemon

    Saídas
    ------
    urls: List<str>
    Lista de urls encontradas

    """
    pokemon = pokedex[pokemon_id]

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
        links += [_img_name_to_url(image["title"]) for image in members]
    return links
