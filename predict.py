#! /usr/bin/python3

import argparse
import json

import cv2  # noqa: I900
import imutils

from qeep.classificador.mobilenet import MobileNetBasic
from qeep.detector.detect_with_classifier import (
    classify_rois,
    filter_detections,
    get_rois,
    read_tuple,
)


def run(image, size="(200, 150)", min_conf=-0.01, visualize=False):

    with open("classes.json", mode="r") as f:
        classes = json.load(f)

    # Parâmetros
    WIDTH = 600
    PYR_SCALE = 1.5
    WIN_STEP = 16
    ROI_SIZE = read_tuple(size, int)

    # Carregamento do modelo
    print("[INFO] Carregando o modelo...")
    model = MobileNetBasic(151)
    model.load(drive=True)  # noqa: E800
    model.model.eval()

    with open("classes.json") as classes_file:
        model.class_names = json.load(classes_file)
    # Carrega a imagem selecionada
    if isinstance(image, str):
        image = cv2.imread(image)
    image = imutils.resize(image, width=WIDTH)

    # Roda o detector
    rois, locs = get_rois(image, PYR_SCALE, WIN_STEP, ROI_SIZE, visualize)

    predictions = classify_rois(model, rois, classes)
    results = filter_detections(image, predictions, locs, min_conf, visualize)

    return results


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

    result = run(args["image"], args["size"], -0.01, -1)

    if args["visualize"]:
        cv2.imshow("Predicoes", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
