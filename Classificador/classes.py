from pathlib import Path
import json
from dataset import loadDataset


def saveClasses(classesPath: Path, datasetPath: Path):
    dataset = loadDataset(datasetPath, None)
    with open(classesPath, mode="w") as f:
        json.dump(dataset.classes, f)


def loadClasses(classesPath: Path):
    with open(classesPath, mode="r") as f:
        classes = json.load(f)

    return classes


if __name__ == "__main__":
    saveClasses(Path("../classes.json"), Path("../data"))
