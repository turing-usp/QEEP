"""
    Scrapper
"""

from typing import List
from abc import ABC, abstractmethod
from pathlib import Path
from hashlib import md5
from functools import cache
from requests import Session
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import filetype


@cache
def resilient_session() -> Session:
    """ previne timeouts """
    session = Session()  # noqa: r2c-requests-use-timeout
    retry = Retry(connect=5, backoff_factor=0.5)
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


class Scrapper(ABC):
    """
    Classe com os metodos basicos para scrappar imagens
    """

    pokemon_id: int
    session: Session

    def __init__(self, pokemon_id: int, session: Session):
        self.pokemon_id = pokemon_id
        self.session = session

    @abstractmethod
    def get_images_url(self) -> List[str]:
        """
        Descrição
        --------
        Descobre todas as imagens de um pokemon

        Saídas
        ------
        urls: List<str>
        Lista de urls encontradas
        """
        return []

    def get_images(self) -> List[bytes]:
        """
        Descrição
        --------
        Descobre baixa todas as imagens pokemons de um site

        Saídas
        ------
        urls: List<bytes>
        Lista de imagens encontradas
        """
        for url in self.get_images_url():
            img = self._download(url)

            if img is None:
                continue

            if not self._is_image(img):
                continue

            process_img = self._img_processing(img)

            yield process_img

    def save(self, base_path: Path):
        """Get images and download in `base_path`"""
        return [self._write_image(base_path, img) for img in self.get_images()]

    @staticmethod
    def _img_processing(img: bytes) -> bytes:
        """
        Descrição
        --------
        Faz algun processamento com a imagen antes de ser salva

        Entradas
        --------
        img: bytes
        Imagem em bytes

        Saídas
        ------
        img: List<bytes>
        Image cortada

        """
        return img

    def _download(self, url: str) -> bytes:
        """
        Descrição
        --------
        Baixa um conteudo

        Entradas
        --------
        url: str
        Url do conteudo

        Saídas
        ------
        conteudo: bytes
        conteudo
        """
        res = self.session.get(url)

        if res is None:
            return None

        if res.status_code != 200:
            return None

        img = res.content

        return img

    @staticmethod
    def _is_image(file) -> bool:
        kind = filetype.guess(file)
        if kind is None:
            return False

        if "image" in kind.mime:
            return True

        return False

    @staticmethod
    def _file_extension(file) -> str:
        kind = filetype.guess(file)
        if kind is None:
            return ""

        return "." + kind.extension

    def _write_image(self, dir_path: Path, img: bytes) -> Path:
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
        assert dir_path.exists()
        assert dir_path.is_dir()

        hashname = md5(img).hexdigest()
        filename = hashname + self._file_extension(img)
        filepath = dir_path / filename

        if filepath.exists():
            return None

        with filepath.open("wb") as file:
            file.write(img)

        return filepath
