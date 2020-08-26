from multiprocessing import Pool
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import bs4
from urllib.parse import quote, unquote
from hashlib import md5
import os
import json
import tqdm

BASE_PATH = os.path.join(os.getcwd(), 'data')


def map_pokemons(gen1=True):
    print("Building Pokédex...")
    pokedex = {}
    pokedex_url = "https://bulbapedia.bulbagarden.net/wiki/List_of_Pok%C3%A9mon_by_National_Pok%C3%A9dex_number"
    req = requests.get(pokedex_url)
    soup = bs4.BeautifulSoup(req.text, features="lxml")
    gens = soup.select("#mw-content-text > table")
    for gen in gens[1:-1]:  # exclui outras tables da wiki
        rows = gen.find_all("tr")
        for row in rows:
            cols = row.find_all("td")
            cols = [item.text.strip() for item in cols]
            try:
                if cols[1]:
                    # formato do dict: pokedex['001']: 'Bulbasaur'
                    pokedex[cols[1].strip('#')] = cols[2]
            except IndexError:
                continue
    if gen1:
        pokedex1 = {}
        i = 0
        for pokemon in pokedex:
            pokedex1[pokemon] = pokedex[pokemon]
            if pokemon == '151':
                break
        print("Pokédex built. Got all 151 original Pokémon.")
        return pokedex1
    print("Pokédex built. Got", len(pokedex), "Pokémon.")
    return pokedex


def scrape_image(link, pokenumber):
    session = requests.Session()
    # prevenir timeout
    retry = Retry(connect=5, backoff_factor=0.5)
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    img_req = session.get(link)
    if img_req.status_code == 200:
        img = img_req.content
        path = os.path.join(BASE_PATH, pokenumber)
        os.makedirs(path, exist_ok=True)
        filename = os.path.join(path, link.split("/")[-1])
        with open(filename, 'wb') as f:
            f.write(img)


def scrape_pokemon(pokemon): # espera no formato ['001', 'Bulbasaur']
    pokenumber = pokemon[0]
    pokename = pokemon[1]
    url = "https://archives.bulbagarden.net/w/api.php"
    params = {
        "action": "query",
        "list": "categorymembers",
        "cmtitle": "Category:" + pokename,
        "cmlimit": "max", # padrão 500
        "cmtype": "file",
        "format": "json",
        "cmcontinue": "" # scrape mais do que cmlimit
    }
    pages = []
    print("Fetching URLs for", pokename)
    while True:  # que perigo
        resp = requests.get(url, params)
        resp = resp.json()
        pages += [image["title"].strip("File:").replace(" ", "_").encode("utf-8")
                  for image in resp["query"]["categorymembers"]]
        if "continue" in resp:
            params["cmcontinue"] = resp["continue"]["cmcontinue"]
            continue
        else:
            break
    # o path real usa md5 do filename pra computar o diretório então dá pra prever. https://www.mediawiki.org/wiki/Manual:$wgHashedUploadDirectory
    links = ["https://archives.bulbagarden.net/media/upload/" + md5(image).hexdigest()[0]
            + "/" + md5(image).hexdigest()[0:2] + "/" + image.decode('utf-8') for image in pages]
    print("Got", len(links), "images for", pokename + ".", "Downloading images...")
    # pool maior aumenta as chances de connection reset, eu acho
    with Pool(12) as p:
        p.starmap(
            scrape_image,
            zip(links, [pokenumber for i in range(len(links))])
        )
    print("Finished downloading images for", pokename + ".")


# sepá dá pra chamar isso com pool mas n sei se acelera mt
for dex, pokemon in map_pokemons().items():
    scrape_pokemon([dex, pokemon])
