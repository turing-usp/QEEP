"""
Site base: https://www.zerochan.net/pikachu?p=2
"""

from typing import List
import requests
import pokebase as pb
from bs4 import BeautifulSoup


def getImagesURLbyId(id: int) -> List[str]:
    print(f"> Pushando #{id} de zerochan.com")

    pokemon = pb.pokemon(id)

    url = f"https://www.zerochan.net/{pokemon.name}"

    page = 0
    links = []
    while True:
        page += 1
        resp = requests.get(f"{url}?p={page}")

        if resp.status_code != 200:
            break

        soup = BeautifulSoup(resp.text, features="lxml")
        imgs = soup.find_all(
            "img", {"alt": pokemon.name.capitalize()})

        if len(imgs) == 0:
            break

        links += [img.get('src') for img in imgs]
    return links


if __name__ == "__main__":
    for id in range(1, 4):
        urls = getImagesURLbyId(id)
        print(len(urls))
        print(*urls, sep="\n")
