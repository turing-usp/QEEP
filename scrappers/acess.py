from typing import List
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import filetype
from hashlib import md5
from pathlib import Path
import os
from functools import cache


def _isImage(file) -> bool:
    kind = filetype.guess(file)
    if kind is None:
        return False

    if "image" in kind.mime:
        return True

    return False


def _fileExtension(file) -> str:
    kind = filetype.guess(file)
    if kind is None:
        return ""

    return "." + kind.extension


@cache
def _resilientSession() -> requests.Session:
    # previne timeouts
    session = requests.Session()
    retry = Retry(connect=5, backoff_factor=0.5)
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session


def createDirIfNotExist(path: Path):
    if not path.exists():
        print("> Create dir:", path)
        os.makedirs(path)


def downloadImgs(urls: List[str]) -> List[bytes]:
    session = _resilientSession()
    imgs_reqs = [session.get(url) for url in urls]
    for img_req in imgs_reqs:
        if img_req is None:
            continue

        if img_req.status_code != 200:
            continue

        img = img_req.content
        if not _isImage(img):
            continue

        yield img


def writeImage(dir_path: Path, img: bytes) -> str:
    """
    Salva a imagem a partir de um diretorio,
    gerando seu nome a partir do hash de seu conteudo
    """
    assert(dir_path.exists())
    assert(dir_path.is_dir())

    hashname = md5(img).hexdigest()
    filename = hashname + _fileExtension(img)
    filepath = dir_path / filename

    if filepath.exists():
        print(">", filepath, "já existe")

    with open(filepath, mode="wb") as f:
        print("> Salvando", filepath)
        f.write(img)

    return filepath


def writeImages(base_path: Path, imgs: List[bytes]):
    return [writeImage(base_path, img) for img in imgs]
