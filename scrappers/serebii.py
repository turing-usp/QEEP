import requests
from bs4 import BeautifulSoup
import os
from uuid import uuid4

ENDPOINT = "https://www.serebii.net"

for id in range(3,151):
    folder = f'dataset/{id}'
    try:
        os.mkdir(folder)
    except FileExistsError:
        pass
    response = requests.get(f"{ENDPOINT}/card/dex/{id:03}.shtml")
    soup = BeautifulSoup(response.text)
    imgs = [s.a.img for s in soup.find_all(class_="cen") if s and s.a and s.a.img]
    links = [ENDPOINT + img.get('src') for img in imgs]
    for index, link in enumerate(links):
        img = requests.get(link).content
        print(f"saving {id}-{index}")
        with open(folder + f'/{index:03}.jpg', mode='wb') as file:
            file.write(img)
