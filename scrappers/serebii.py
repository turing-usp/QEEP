"""
Site base: https://www.serebii.net
"""

from typing import List
import requests
import re
from bs4 import BeautifulSoup


def getImagesURLbyId(id: int) -> List[str]:
    print(f"> Pushando #{id} de serebii.net")

    url = f"https://www.serebii.net/card/dex/{id:03}.shtml"

    response = requests.get(url)
    soup = BeautifulSoup(response.text, features="lxml")
    imgs = soup.find_all("img", {"src": re.compile("/card/th/.*.jpg")})
    links = ["https://www.serebii.net" + img.get('src') for img in imgs]

    artURL = f"https://www.serebii.net/art/th/{id}.png"
    return [artURL] + links


if __name__ == "__main__":
    for id in range(1, 3):
        print(getImagesURLbyId(id))
