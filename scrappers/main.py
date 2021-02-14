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
    imgsURL = getAllImagesURLbyId(1)
    imgs = acess.downloadImgs(imgsURL)

    p = Path('../data')
    acess.createDirIfNotExist(p)
    for img in imgs:
        acess.writeImage(p, img)
