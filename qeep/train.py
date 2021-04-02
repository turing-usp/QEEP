import argparse
import sys
from classificador.mobilenet import run


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Treina na mobilet")

    parser.add_argument(
        "--show",
        dest="show",
        default=False,
        type=bool,
        help="Mostra a rede",
    )

    parser.add_argument(
        "--load",
        dest="load",
        default=True,
        type=bool,
        help="Carrega os pesos do modelo",
    )

    parser.add_argument(
        "--input",
        dest="img_path",
        type=str,
        help="Imagem que servirá de entrada para o modelo",
    )

    parser.add_argument(
        "--classes",
        dest="class_names_path",
        default="./classes.json",
        type=str,
        help="Path do json com as classes",
    )

    parser.add_argument(
        "--train",
        dest="is_train",
        default=False,
        type=bool,
        help="Define que o modelo deve ser treinado",
    )

    parser.add_argument(
        "--dataset-path",
        dest="dataset_path",
        default="./data",
        type=str,
        help="Local em que o dataset será carregado, caso não sejá encontrado será baixado",
    )

    parser.add_argument(
        "--dataset-url-id",
        dest="dataset_url_id",
        default="1SA7wV7BwEpNoR721aUSauFvqCTfXba1h",
        type=str,
        help="URL do drive na qual o dataset será baixado",
    )

    parser.add_argument(
        "--dataloader-batch-size",
        dest="dataloader_batch_size",
        default=4,
        type=int,
        help="Bathsize do dataloader",
    )

    parser.add_argument(
        "--dataloader-num-workers",
        dest="dataloader_num_workers",
        default=4,
        type=int,
        help="Numero de workrs do dataloader",
    )

    parser.add_argument(
        "--dataloader-shuffle",
        dest="dataloader_shuffle",
        default=True,
        type=bool,
        help="Embraralhar dataloader",
    )

    parser.add_argument(
        "--dataset-tresh-hold",
        dest="dataset_tresh_hold",
        default=0.8,
        type=float,
        help="Porcentagem do dataset usado para treino",
    )

    parser.add_argument(
        "--optimizer-learning-rate",
        dest="optimizer_learning_rate",
        default=0.001,
        type=float,
        help="Learning rate do otimizador",
    )

    parser.add_argument(
        "--optimizer-momentum",
        dest="optimizer_momentum",
        default=0.9,
        type=float,
        help="Momentum do otimizador",
    )

    parser.add_argument(
        "--scheduler-step-size",
        dest="scheduler_step_size",
        default=7,
        type=int,
        help="Step size do schedule",
    )

    parser.add_argument(
        "--scheduler-gamma",
        dest="scheduler_gamma",
        default=0.1,
        type=float,
        help="Step size do schedule",
    )

    parser.add_argument(
        "--epochs",
        dest="epochs",
        default=25,
        type=int,
        help="Numero de epocas treinadas",
    )

    parser.add_argument(
        "--save",
        dest="save",
        default="mobilenet_weight.pkl",
        type=str,
        help="Para salvar o modelo",
    )

    parse_args = parser.parse_args()

    if (not parse_args.is_train) and (parse_args.img_path is None):
        print("input image is required to run")
        sys.exit(1)

    run(parse_args)
