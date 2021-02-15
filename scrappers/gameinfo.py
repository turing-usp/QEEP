"""
Site base: https://gameinfo.io
"""

from typing import List


def getImagesURLbyId(id: int) -> List[str]:
    print(f"> Pushando #{id} de gamainfo.io")

    normalImg = f"https://images.gameinfo.io/pokemon/256/{id:03}-00.png"
    shineImg = f"https://images.gameinfo.io/pokemon/256/{id:03}-00-shiny.png"
    return [normalImg, shineImg]


if __name__ == "__main__":
    for id in range(1, 3):
        print(getImagesURLbyId(id))
