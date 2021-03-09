from typing import List
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import filetype
from hashlib import md5
from pathlib import Path
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


def downloadImg(url: str) -> bytes:
    """
    Descrição
    --------
    Baixa uma imagens

    Entradas
    --------
    url: str
    Url da imagem

    Saídas
    ------
    images: List<bytes>
    Lista de imagens

    """
    session = _resilientSession()

    img_req = session.get(url)

    if img_req is None:
        return

    if img_req.status_code != 200:
        return

    img = img_req.content
    if not _isImage(img):
        return

    return img


def downloadImgs(urls: List[str]) -> List[bytes]:
    """
    Descrição
    --------
    Baixa uma lista de imagens

    Entradas
    --------
    urls: List<str>
    Lista de urls

    Saídas
    ------
    images: List<bytes>
    Lista de imagens

    """

    print(f"> Baixando {len(urls)} imagens...")
    for url in urls:
        img = downloadImg(url)

        if img is None:
            continue

        yield img


def writeImage(dir_path: Path, img: bytes) -> Path:
    """
    Descrição
    --------
    Salva a imagem a partir de um diretorio,
    gerando seu nome a partir do hash de seu conteudo
    evitando assim salvar duas imagens iguais

    Entradas
    --------
    dir_path: Path
    Diretorio em que a imagem será salva

    img: bytes
    Imagem que será salva

    Saídas
    ------
    filepath: Path
    local onde a imagem foi salva

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
    """
    Descrição
    --------
    Salva uma lista de imagem a partir de um diretorio,
    gerando seus nomes a partir do hash de seu conteudo
    evitando assim salvar duas imagens iguais

    Entradas
    --------
    dir_path: Path
    Diretorio em que a imagem será salva

    img: List<bytes>
    Imagens que serão salvas

    Saídas
    ------
    filepaths
    Lista de onde as iamgens foram salvas

    """
    return [writeImage(base_path, img) for img in imgs]
