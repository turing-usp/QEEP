from itertools import chain
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms, datasets
import torchvision
from pathlib import Path

DATASET_PATH = Path('../data')

torch.manual_seed(4)

data_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def loadDataset(path: Path, tranform: torch.nn.Module):
    if not path.exists():
        raise Exception('Dataset not found')

    return datasets.ImageFolder(root=path,
                                transform=tranform)


def loadSplitedDataset(tresh_hold: float, path: Path, transform: torch.nn.Module):
    dataset = loadDataset(path, transform)
    split_dataset = torch.utils.data.random_split(
        dataset, [int(len(dataset) * tresh_hold), int(len(dataset) * (1-tresh_hold))])
    return split_dataset[0], split_dataset[1], dataset


def loadSplitedLoader(batch_size: int, num_workers: int, tresh_hold: float, path: Path, transform: torch.nn.Module):
    train_dataset, val_dataset, datasets= loadSplitedDataset(tresh_hold, path, transform)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return train_loader, val_loader, datasets


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

