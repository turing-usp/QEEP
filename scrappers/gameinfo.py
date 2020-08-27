from util import getImg
import asyncio
from uuid import uuid4
import os

async def main():
    for id in range(1, 151):
        folder = f"dataset/{id:03}"
        url = f"https://images.gameinfo.io/pokemon/256/{id:03}-00.png"

        try:
            os.mkdir(folder)
        except FileExistsError:
            pass
        filename = folder + f"/{uuid4()}.png"
        print(filename)
        getImg(url, filename)

if __name__ == "__main__":
    asyncio.run(main())
