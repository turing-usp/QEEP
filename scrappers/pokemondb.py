import requests
from bs4 import BeautifulSoup
import os
from util import getImg
import re
from uuid import uuid4

ENDPOINT = "https://www.pokemondb.net"

def main():
    for id, url in getPokemons().items():
        folder = f'dataset/{id:03}'
        scrapper_link(folder, url, id)

def getPokemons():
    url = f"{ENDPOINT}/pokedex/stats/combo"
    response = requests.get(url)
    soup = BeautifulSoup(response.text)

    hrefs = soup.find("table").find_all("a", {"href": re.compile("/pokedex/")})

    links = {}
    count = 1
    for href in hrefs:
        link = ENDPOINT + href.get('href')
        if count >= 151:
            break
        if link not in links.values():
            links[count] = link
            count+=1

    return links

def scrapper_link(folder, url, id):
    try:
        os.mkdir(folder)
    except FileExistsError:
        pass

    response = requests.get(url)
    soup = BeautifulSoup(response.text)

    pattern = r"https://img.pokemondb.net/sprites/[\w-]*/normal/[\w \.]*"
    imgs = soup.find_all("img", {"src": re.compile(pattern)})

    links = [img.get('src') for img in imgs]

    for link in links:
        extension = link.split(".")[-1]
        filename = folder + f"/{uuid4()}.{extension}"
        print(filename)
        getImg(link, filename)

if __name__ == "__main__":
   main() 

