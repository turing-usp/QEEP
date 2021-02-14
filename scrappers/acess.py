from typing import List
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import filetype
from hashlib import md5
from pathlib import Path
import os


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


def createDirIfNotExist(path: Path):
    if not path.exists():
        os.makedirs(path)


def downloadImgs(urls: List[str]) -> List[bytes]:
    for url in urls:
        session = requests.Session()
        # previne timeouts
        retry = Retry(connect=5, backoff_factor=0.5)
        adapter = HTTPAdapter(max_retries=retry)
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        img_req = session.get(url)

        if img_req.status_code != 200:
            continue

        img = img_req.content
        if not _isImage(img):
            continue

        yield img


def writeImage(path: Path, img: bytes):
    assert(path.exists())
    assert(path.is_dir())

    hashname = md5(img).hexdigest()
    filename = hashname + _fileExtension(img)
    filepath = path / filename

    if filepath.exists():
        print(filepath, "já existe")

    with open(filepath, mode="wb") as f:
        f.write(img)
