from pathlib import Path
import argparse
from qeep.scrapper.all import get_all_images_and_save_by_ids

parser = argparse.ArgumentParser(
    description="Baixa imagens de pokemons em um diretorio"
)
parser.add_argument(
    "-b", "--begin", default=1, type=int, help="pokemon inicial"
)
parser.add_argument(
    "-e", "--end", default=151, type=int, help="ultimo pokemon a ser procurado"
)
parser.add_argument(
    "-p", "--path", default="./data", type=str, help="diretorio que será salvo"
)

args = parser.parse_args()
path = Path(args.path)
get_all_images_and_save_by_ids(range(args.begin, args.end + 1), path)
