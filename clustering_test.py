from sklearn.cluster import KMeans
import random
import torchvision
import torch
import torch.nn as nn
import skimage
from torch.utils.data import Dataset
from typing import Callable
from collections import OrderedDict
from frame import render_polygon
from functools import partial
import torchvision.transforms as transforms
from torch.nn import functional as F
import flow_vis
import numpy as np
import matplotlib.pyplot as plt


class CustomDataset(Dataset):
    def __init__(self, data, transforms=None):
        self.data = data
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]

        if self.transforms != None:
            image = self.transforms(image)
        return image


# # # ### Train


def build_codebook(inputs):
    samples = []
    for t in inputs:
        for i in range(0, 8):
            for j in range(0, 8):
                x = t[:, i * 16 : i * 16 + 16, j * 16 : j * 16 + 16]
                x = x.detach().numpy().reshape(768)
                samples.append(x)
    final_samples = np.array(samples)

    kmeans = KMeans(n_clusters=64, random_state=0, n_init="auto").fit(final_samples)
    return kmeans.cluster_centers_


def reconstruct_from_codebooks(t, codebooks):
    image = np.zeros((3, 128, 128))
    for i in range(0, 8):
        for j in range(0, 8):
            for patches in codebooks:
                x = t[:, i * 16 : i * 16 + 16, j * 16 : j * 16 + 16]
                x = x.detach().numpy().reshape(768)
                x = x.reshape(-1, 1)
                distances = np.sqrt(np.sum((patches.T - x) ** 2, axis=0))
                patchidx = np.argmin(distances)
                image[:, i * 16 : i * 16 + 16, j * 16 : j * 16 + 16] += patches[
                    patchidx
                ].reshape(3, 16, 16)
    return image


albedo_images = []
for i in range(0, 1000):
    albedo = f"data/example_{i}_normal0001.png"
    albedo_images.append(skimage.io.imread(albedo))

curr_transforms = [
    transforms.ToTensor(),
]

inputs = CustomDataset(
    albedo_images,
    transforms=transforms.Compose(curr_transforms),
)


def get_codebooks(inputs, layers=4):
    codebook = build_codebook(inputs)
    codebooks = [codebook]
    for i in range(0, layers):
        next_images = []
        for t in inputs:
            image = reconstruct_from_codebooks(t, codebooks)
            next_images.append((image - t.detach().numpy()).T)

        inputs2 = CustomDataset(
            next_images,
            transforms=transforms.Compose(curr_transforms),
        )
        codebook2 = build_codebook(inputs2)
        codebooks.append(codebook2)
    return codebooks


t = inputs[12]

codebooks = get_codebooks(inputs, 2)
image = reconstruct_from_codebooks(t, codebooks)

fig, ax = plt.subplots(1, 3, figsize=(6, 6))

ax[0].imshow(image.T, cmap="gray")
ax[1].imshow(t.detach().numpy().T, cmap="gray")
ax[2].imshow(image.T - t.detach().numpy().T, cmap="gray")
plt.tight_layout()
plt.show()
