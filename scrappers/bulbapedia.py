import os
import requests
import bs4
from multiprocessing import Pool
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from hashlib import md5

BASE_PATH = os.path.join(os.getcwd(), 'data')


def map_pokemons(start=1, end=151, all=False):
    """
    Constrói um dicionário com Pokémon e seus números da Pokédex, de acordo com a Bulbapedia.
    O número da Pokédex é padronizado com zeros à esquerda.
    Por padrão, contém apenas a primeira geração de Pokémon (1-151).
    Pode parar de funcionar caso a estrutura da Bulbapedia mude.

    Parâmetros:
        start (int): número do primeiro Pokémon. Padrão: 1
        end (int): número do último Pokémon (inclusivo). Padrão: 151
        all (bool): se todos os Pokémon devem ser inclusos. Sobrepões start/end. Padrão: False
    """
    print("Construindo Pokédex...")
    pokedex = {}
    pokedex_url = "https://bulbapedia.bulbagarden.net/wiki/List_of_Pok%C3%A9mon_by_National_Pok%C3%A9dex_number"
    req = requests.get(pokedex_url)
    soup = bs4.BeautifulSoup(req.text, features="lxml")
    gens = soup.select("#mw-content-text > table")
    for gen in gens[1:-1]:  # exclui tabelas inúteis
        rows = gen.find_all("tr")
        for row in rows:
            cols = row.find_all("td")
            cols = [item.text.strip() for item in cols]
            try:
                if cols[1]:
                    # formato: pokedex['001']: 'Bulbasaur'
                    pokedex[cols[1].strip('#')] = cols[2]
            except IndexError:
                continue
    if len(pokedex) > end and not all:
        pokedex_trimmed = {}
        for pokemon in pokedex:
            try:
                if int(pokemon) < start:
                    pass
                else:
                    pokedex_trimmed[pokemon] = pokedex[pokemon]
                if int(pokemon) == end:
                    break
            except ValueError:
                pass
        print("Pokédex built. Got", len(pokedex_trimmed), "Pokémon.")
        return pokedex_trimmed
    print("Full Pokédex built. Got", len(pokedex), "Pokémon.")
    return pokedex


def scrape_image(url: str, pokenumber: str):
    """
    Baixa uma única imagem e a salva em uma pasta definida pelo número da Pokédex.
    A imagem é salva para a pasta 'BASE_PATH/pokenumber'.

    Parâmetros:
        url (str): url da imagem a ser baixada.
        pokenumber (str): número da Pokédex, pré-formatado (###).
    """
    session = requests.Session()
    # previne timeouts
    retry = Retry(connect=5, backoff_factor=0.5)
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    img_req = session.get(url)
    if img_req.status_code == 200:
        img = img_req.content
        path = os.path.join(BASE_PATH, pokenumber)
        os.makedirs(path, exist_ok=True)
        filename = os.path.join(path, url.split("/")[-1])
        with open(filename, 'wb') as f:
            f.write(img)


def scrape_pokemon(pokemon: list):
    """
    Busca todas as imagens de um Pokémon na Bulbapedia e chama scrape_image() para baixar e salvar.
    Recebe um único parâmetro 'pokemon' no formato ["número", "nome"], gerado por map_pokemons().
    """
    try:
        pokenumber = pokemon[0]
        pokename = pokemon[1]
    except IndexError as e:
        print("Erro: índice inválido. O formato do argumento é [\"001\", \"Bulbasaur\"]")
    url = "https://archives.bulbagarden.net/w/api.php"
    params = {
        "action": "query",
        "list": "categorymembers",
        "cmtitle": "Category:" + pokename,
        "cmlimit": "max",  # padrão 500
        "cmtype": "file",
        "format": "json",
        "cmcontinue": ""  # continua o scraping caso haja mais imagens do que o cmlimit
    }
    pages = []
    print("Coletando URLs para", pokename)
    while True:  #  perigoso mas não tem do-while em Python
        resp = requests.get(url, params)
        resp = resp.json()
        pages += [image["title"].strip("File:").replace(" ", "_").encode("utf-8") for image in resp["query"]["categorymembers"]]
        if "continue" in resp:
            params["cmcontinue"] = resp["continue"]["cmcontinue"]
            continue
        else:
            break
    # a estrutura do diretório da imagem é composto usando o hash md5 do nome do arquivo.
    # Documentado em: https://www.mediawiki.org/wiki/Manual:$wgHashedUploadDirectory
    links = ["https://archives.bulbagarden.net/media/upload/" + md5(image).hexdigest()[0] + "/" + md5(image).hexdigest()[0:2] + "/" + image.decode('utf-8') for image in pages]
    print(len(links), "imagens obtidas para", pokename + ".", "Baixando imagens...")
    # Acho que aumentar o pool aumenta as chances de timeout da conexão. Não tenho certeza.
    with Pool(os.cpu_count() - 1) as p:
        p.starmap(
            scrape_image,
            zip(links, [pokenumber for i in range(len(links))])
        )
    print("Imagens para", pokename, "baixadas.")


pokedex = map_pokemons()
# talvez funcione usando multiprocessing.Pool mas não sei se faria diferença.
for dex, pokemon in pokedex.items():
    scrape_pokemon([dex, pokemon])
