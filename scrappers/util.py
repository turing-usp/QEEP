# Scrapper to oficial pokedex

import requests

def saveImg(img_data, filename):
    with open(filename, mode='wb') as file:
        file.write(img_data)

def getImg(url, filename):
    img_data = requests.get(url).content
    saveImg(img_data, filename)
