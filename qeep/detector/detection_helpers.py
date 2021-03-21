# import the necessary packages
import imutils
import numpy as np
import PIL as Image


def sliding_window(image: Image, step: int, ws: tuple) -> Image:
    """
    Descrição
    ---------
    Passa a janela deslizante na imagem.

    Entradas
    --------
    image: (PIL Image)
    Imagem a ser tratada

    step: (int)
    passos, em pixels, da janela deslizante

    ws (tuple)
    formato da janela

    Saidas
    ------
    A janela atual (Image)
    """
    # passa a janela pela imagem
    for y in range(0, image.shape[0] - ws[1], step):
        for x in range(0, image.shape[1] - ws[0], step):
            # yield a janela atual
            yield (x, y, image[y : y + ws[1], x : x + ws[0]])  # noqa: E203


def image_pyramid(
    image: Image, scale: float = 1.5, min_size: tuple = (224, 224)
) -> Image:
    """
    Descrição
    ---------
    Aplica o algoritmo de piramide na imagem

    Entradas
    --------
    image: (PIL Image)
    Imagem a ser tratada

    scale (float)
    fator de redução da escala.

    minsize (tuple)
    tamanho minimo da imagem de saida, serve como ponto de parada

    Saidas
    ------
    A imagem para passar as janelas deslizantes (Image)
    """
    # yield a imagem original
    yield image
    # loop para pegar todas as imagens da piramide
    while True:
        # computa o tamanho da proxima imagem
        w = int(image.shape[1] / scale)
        image = imutils.resize(image, width=w)
        # condição de parada: se a imagem ficar menor que o tamanho minimo permitido
        if image.shape[0] < min_size[1] or image.shape[1] < min_size[0]:
            break
        # yield the next image in the pyramid
        yield image


def img_to_array(img: Image) -> np.array:
    """
    Descrição
    ---------
    Converte uma PIL Image em um Numpy array.

    Entradas
    --------
    img (Image)

    Saidas
    ------
    x (mp.array)
    A imagem em Numpy array
    """
    x = np.asarray(img, dtype="int8")
    if len(x.shape) == 2:
        x = x.reshape((x.shape[0], x.shape[1], 1))
    elif len(x.shape) != 3:
        raise ValueError(f"Unsupported image shape: {x.shape}")
    return x
