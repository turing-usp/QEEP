from util import getImg
import asyncio
from uuid import uuid4
import os

async def main():
    for id in range(1, 151):
        folder = f"dataset/{id}"
        url = f"https://assets.pokemon.com/assets/cms2/img/pokedex/full/{id:03}.png"

        try:
            os.mkdir(folder)
        except FileExistsError:
            pass
        filename = folder + f"/{uuid4()}.png"
        print(filename)
        await getImg(url, filename)

if __name__ == "__main__":
    asyncio.run(main())
