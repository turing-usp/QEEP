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
pokeDataset = datasets.ImageFolder(root=DATASET_PATH,
                                   transform=data_transform)

dataloader = torch.utils.data.DataLoader(pokeDataset,
                                         batch_size=4, shuffle=True,
                                         num_workers=4)


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
    for inputs, classes in dataloader:
        out = torchvision.utils.make_grid(inputs)
        imshow(out, title=[pokeDataset.classes[x] for x in classes])
        plt.show()
