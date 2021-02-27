from itertools import chain
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms, datasets
import torchvision

DATASET_PATH = '../data'

data_transform = transforms.Compose([
    transforms.Resize((124, 124)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

_pokeDataset_full = datasets.ImageFolder(root=DATASET_PATH,
                                         transform=data_transform)

datasetLen = {
    'full': len(_pokeDataset_full)
}
datasetLen['train'] = int(0.7*datasetLen['full'])
datasetLen['val'] = datasetLen['full'] - datasetLen['train']

[_pokeDataset_train, _pokeDataset_val] = torch.utils.data.random_split(
    _pokeDataset_full, [datasetLen['train'], datasetLen['val']])

pokeDataset = {
    'full': _pokeDataset_full,
    'train': _pokeDataset_train,
    'val': _pokeDataset_val
}

pokeLoader = {
    'train': torch.utils.data.DataLoader(pokeDataset['train'],
                                         batch_size=4, shuffle=True,
                                         num_workers=4),
    'val': torch.utils.data.DataLoader(pokeDataset['val'],
                                       batch_size=4, shuffle=True,
                                       num_workers=4)
}
pokeLoader['full'] = chain(pokeLoader['train'], pokeLoader['val'])


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


if __name__ == "__main__":
    # Get a batch of training data
    for inputs, classes in (pokeLoader['full']):
        out = torchvision.utils.make_grid(inputs)
        imshow(out, title=[_pokeDataset_full.classes[x] for x in classes])
        plt.show()
