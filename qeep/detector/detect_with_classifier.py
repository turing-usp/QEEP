# Referência:
# Turning any CNN image classifier into an object detector with Keras, TensorFlow, and OpenCV
# https://www.pyimagesearch.com/2020/06/22/turning-any-cnn-image-classifier-into-an-object-detector-with-keras-tensorflow-and-opencv/

# Conversão de uma CNN classificadora em um detector de objetos com Pytorch

import json
import argparse
import time
from typing import Type, Tuple, Callable, List
import imutils
import cv2  # noqa: I900
import torch
import numpy as np
import PIL as Image
from imutils.object_detection import non_max_suppression
from .detection_helpers import (
    img_to_array,
    sliding_window,
    image_pyramid,
)
from classificador.mobilenet import MobileNet


def get_rois(
    original_img: Image,
    PYR_SCALE: float,
    WIN_STEP: int,
    ROI_SIZE: tuple,
    visualize: int,
) -> Tuple[List[np.array], List[Tuple[float, float, float, float]]]:
    """
    Descrição
    ---------
    Itera sobre o metodo da piramide para gerar as regiões de interesse

    Entradas
    --------
    original_img (Image)
    Imagem de entrada

    PYR_SCALE (float)
    escala usada no metodo da piramide.
    WIN_STEP (int)
    passo entre cada window.

    ROI_SIZE (float)
    tamanho da janela.

    visualize (int)
    se > 0 entra em modo debug e visualiza cada janela na imagem original

    Saidas
    ------
    rois (tuple)
    regiões de interesse a serem classificadas

    locs (tuple)
    localização de cada roi na imagem original
    """
    # Inicializa a pirâmide da imagem
    pyramid = image_pyramid(original_img, scale=PYR_SCALE, min_size=ROI_SIZE)
    # Inicializa a lista de ROIs (Regiões de Interesse) geradas pela
    # pirâmide e pela sliding window, e a lista de coordenadas (x, y)
    # para guardar a posição de cada ROI na imagem original
    rois = []
    locs = []
    # Tempo inicial do pré-processamento da imagem original
    start = time.time()

    # Dimensões da imagem original
    (_, W) = original_img.shape[:2]

    # Aplica o algoritmo da pirâmide sobre a imagem
    for image in pyramid:
        # Obtém o fator de escala entre a imagem original e
        # a camada atual da pirâmide
        scale = W / float(image.shape[1])
        # Para cada camada da pirâmide, aplica o algoritmo de sliding window
        for (x, y, original_roi) in sliding_window(image, WIN_STEP, ROI_SIZE):
            # Reescala as coordenadas (x, y) da ROI com respeito às
            # dimensões da imagem original
            x = int(x * scale)
            y = int(y * scale)
            w = int(ROI_SIZE[0] * scale)
            h = int(ROI_SIZE[1] * scale)
            # Transforma a ROI (PIL Image) em array para a classificação
            roi = img_to_array(original_roi)
            # Atualiza a lista de ROIS e a lista de coordenadas
            rois.append(roi)
            locs.append((x, y, x + w, y + h))

            # Checa se as sliding windows devem ser mostradas
            if visualize > 0:
                # Clona a imagem original e então desenha as bounding boxes
                # representando a ROI atual
                clone = original_img.copy()
                cv2.rectangle(clone, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # Mostra a ROI atual na tela
                cv2.imshow("Sliding Window", clone)
                cv2.imshow("ROI", original_roi)
                cv2.waitKey(0)

    # Fecha as sliding windows mostradas
    if visualize > 0:
        cv2.destroyAllWindows()

    # Tempo final do pré-processamento da imagem original
    end = time.time()
    # Duração do pré-processamento da imagem original
    print(
        f"[INFO] O pré-processamento da imagem durou {end - start:.5f} segundos"
    )

    return rois, locs


def classify_rois(
    model: torch.nn.Module, rois: np.array, classes: List[str]
) -> List[Tuple[float, float, float, float]]:
    """
    Descrição
    ---------
    Classifica as ROIs

    Entradas
    --------
    model (torch.nn.Module)
    modelo
    rois (np.array)
    regioes a serem classificadas.
    classes (List[str])
    lista com as possiveis classes.

    Saidas
    ------
    A imagem para passar as janelas deslizantes (List[Tuple[float, float, float, float]])
    """
    print("[INFO] Classificando as ROIs...")
    # Tempo inicial da classificação das ROIs
    start = time.time()

    # Classifica cada uma das ROIs utilizando o modelo
    preds = [model.predict(img)[0] for img in rois]

    # Transforma as predições em um array
    preds = list(map(lambda x: x.detach().cpu().numpy(), preds))
    preds = np.array(preds)

    # Tempo final da classificação das ROIs
    end = time.time()
    print(f"[INFO] A classificalão das ROIs durou {end - start:.5f} segundos")

    # Inicializa a lista de predições das ROIs
    results = []
    # Itera sobre todas as predições
    for pred in preds:
        # Seleciona o índice da classe mais provável
        top_indice = pred.argsort()[0, -1]
        # Cria uma tupla contendo top_indice, a classe e a probabilidade
        result = (top_indice, classes[top_indice], pred[0][top_indice])
        # Adiciona a tupla criada na lista de resultados
        results.append(result)

    return results


def filter_detections(
    original_img: Image,
    preds: List[tuple],
    locs: np.array,
    min_conf: float,
    visualize: int,
) -> Type[None]:
    """
    Definição
    ---------
    Mostra as predições

    Entradas
    --------
    original_img (Image)
    imagem original.

    preds (List[tuple])
    lista com as prediçoes (indice, classe, probabilidade).

    locs (np.array)
    localização das janelas classificadas

    min_conf (float)
    probabilidade minima para se considerar uma detecçao valida

    visualize (int)
    variavel de debub, se >0 permite visualizar passos intermediarios do processo
    """
    # Inicializa dicionário de predições separado por classes

    labels = {}

    # Itera sobre todas as predições de cada ROI
    for (i, p) in enumerate(preds):
        # Separa as informações da ROI atual
        (_, label, prob) = p
        # Filtra as detecções com baixa prioridade de acordo com min_conf
        if prob >= min_conf:
            # Seleciona a bounding box da predição
            box = locs[i]
            # Caso a chave label exista no dicionário labels, guarda seu valor em lab
            # Se não, guarda uma lista vazia em lab
            lab = labels.get(label, [])
            # Adiciona a bounding box e sua probabilidade no dicionário de labels
            lab.append((box, prob))
            labels[label] = lab

    # Itera sobre cada uma das classes detectadas
    for label in labels:
        print(f"[INFO] Apresentando os resutados para '{label}'")

        # Checa se as bounding boxes encontradas devem ser mostradas
        if visualize > 0:
            # Clona a imagem original para desenhar as bounding boxes
            clone = original_img.copy()
            # Itera sobre todas as bounding boxes da classe atual
            for (box, _) in labels[label]:
                # Desenha as bounding bouxes na imagem
                (startX, startY, endX, endY) = box
                cv2.rectangle(
                    clone, (startX, startY), (endX, endY), (0, 255, 0), 2
                )
            cv2.imshow("Debug", clone)

        # Clona a imagem original para desenhar as bounding boxes
        clone = original_img.copy()
        # Aplica a técnica de non-maxima supression nas predições da classe atual
        boxes = np.array([p[0] for p in labels[label]])
        proba = np.array([p[1] for p in labels[label]])
        boxes = non_max_suppression(boxes, proba)
        # Itera sobre as bounding boxes remanescenes
        for (startX, startY, endX, endY) in boxes:
            # Desenha as bounding bouxes na imagem
            cv2.rectangle(
                clone, (startX, startY), (endX, endY), (0, 255, 0), 2
            )
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.putText(
                clone,
                label,
                (startX, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 0),
                2,
            )
        cv2.imshow("Predicoes", clone)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def read_tuple(text: str, cast_function: Callable = str) -> tuple:
    """
    Lê uma string que tem uma tupla, convertendo os parametros pela função de cast.
    ex: `read_tuple('(1, 2)', int) == (1,2)`
    """
    without_brackets = text.strip("()")
    values = [cast_function(i) for i in without_brackets.split(",")]
    return tuple(values)


if __name__ == "__main__":

    # Arg parser
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-i", "--image", required=True, help="path to the input image"
    )
    ap.add_argument(
        "-s",
        "--size",
        type=str,
        default="(200, 150)",
        help="ROI size (in pixels)",
    )
    ap.add_argument(
        "-c",
        "--min-conf",
        type=float,
        default=0.9,
        help="minimum probability to filter weak detections",
    )
    ap.add_argument(
        "-v",
        "--visualize",
        type=int,
        default=-1,
        help="whether or not to show extra visualizations for debugging",
    )
    args = vars(ap.parse_args())

    with open("classes.json", mode="r") as f:
        classes = json.load(f)

    # Parâmetros
    WIDTH = 600
    PYR_SCALE = 1.5
    WIN_STEP = 16
    ROI_SIZE = read_tuple(args["size"], int)

    # Carregamento do modelo
    print("[INFO] Carregando o modelo...")
    model = MobileNet(151)
    # model.loadModel() # noqa: E800
    model = model.model.eval()
    # Carrega a imagem selecionada
    image = cv2.imread(args["image"])
    image = imutils.resize(image, width=WIDTH)

    # Roda o detector
    rois, locs = get_rois(
        image, PYR_SCALE, WIN_STEP, ROI_SIZE, args["visualize"]
    )
    predictions = classify_rois(model, rois, classes)
    filter_detections(
        image, predictions, locs, args["min_conf"], args["visualize"]
    )
