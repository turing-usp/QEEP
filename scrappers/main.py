from typing import List
from pathlib import Path

import bulbapedia
import gameinfo
import pokemondb
import pokemon
import serebii

import acess


def getAllImagesURLbyId(id: int) -> List[str]:
    acc = []
    acc += pokemon.getImagesURLbyId(id)
    acc += gameinfo.getImagesURLbyId(id)
    acc += serebii.getImagesURLbyId(id)
    acc += pokemondb.getImagesURLbyId(id)
    acc += bulbapedia.getImagesURLbyId(id)
    return acc


if __name__ == "__main__":
    imgs = ["https://images.gameinfo.io/pokemon/256/001-00-shiny.webp"]
    bin_img = list(acess.downloadImgs(imgs))[0]

    p = Path('../data')
    acess.createDirIfNotExist(p)
    acess.writeImage(p, bin_img)
