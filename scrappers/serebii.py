import requests
from bs4 import BeautifulSoup
import os
from util import getImg
import asyncio
import re
from uuid import uuid4

ENDPOINT = "https://www.serebii.net"

async def main():
    for id in range(1,151):
        folder = f'dataset/{id}'
        url = f"{ENDPOINT}/card/dex/{id:03}.shtml"
        await scrapper_link(folder, url)

async def scrapper_link(folder, url):
    try:
        os.mkdir(folder)
    except FileExistsError:
        pass

    response = requests.get(url)
    soup = BeautifulSoup(response.text)

    imgs = soup.find_all("img", {"src": re.compile("/card/th/.*.jpg")})

    links = [ENDPOINT + img.get('src') for img in imgs]

    for i, link in enumerate(links):
        filename = folder + f"/{uuid4()}.jpg"
        print(filename)
        await getImg(link, filename)

if __name__ == "__main__":
   asyncio.run(main()) 

