"""
Site base: https://pokemon.com
"""

from typing import List


def getImagesURLbyId(id: int) -> List[str]:
    print(f"> Pushando #{id} de pokemon.com")

    return [f"https://assets.pokemon.com/assets/cms2/img/pokedex/full/{id:03}.png"]


if __name__ == "__main__":
    for id in range(1, 3):
        print(getImagesURLbyId(id))
