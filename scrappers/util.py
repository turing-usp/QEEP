# Scrapper to oficial pokedex

import requests
import asyncio

async def saveImg(img_data, filename):
    with open(filename, mode='wb') as file:
        file.write(img_data)

async def getImg(url, filename):
    img_data = requests.get(url).content
    await saveImg(img_data, filename)
