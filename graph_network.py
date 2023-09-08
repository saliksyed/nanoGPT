import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import make_grid

from conv_autoencoder import (
    CustomDataset,
    ConvolutionalAutoencoder,
    Autoencoder,
    Encoder,
    Decoder,
)

from frame import render_polygon
import matplotlib.pyplot as plt

from config import WEIGHTS_DIR, N_SIDES

if not os.path.exists(WEIGHTS_DIR):
    os.mkdir(WEIGHTS_DIR)


class Edge:
    def __init__(self, weight, node):
        self.weight = weight
        self.node = node


class Node:
    def __init__(self, id, crop):
        self.model = ConvolutionalAutoencoder(
            Autoencoder(Encoder(), Decoder()),
            checkpoint_path=os.path.join(WEIGHTS_DIR, id + ".pt"),
        )
        self.crop = crop
        self.incoming_edges = []
        self.outgoing_edges = []
        self.data = []
        self.current = None

    def clear(self):
        self.current = None
        for child in self.outgoing_edges:
            child.node.clear()

    def load_weights(self):
        self.model.load_weights()

    def generate(self, example):
        if self.current != None:
            return self.current
        # crop and normalize the example
        d = CustomDataset(
            [example],
            transforms=transforms.Compose(
                [
                    transforms.ToTensor(),
                    lambda img: transforms.functional.crop(img, *self.crop),
                    lambda img: transforms.functional.resize(
                        img,
                        (32, 32),
                        antialias=True,
                    ),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            ),
        )
        # make prediction
        base_prediction = self.model.generate(d[0]).cpu()
        child_results = []
        for child in self.incoming_edges:
            child_results.append(child, child.node.generate(example))
        # resize the output
        for child, result in child_results:
            x = child.node.crop[0]
            y = child.node.crop[1]
            w = child.node.crop[2]
            h = child.node.crop[3]
            base_prediction[:, x : x + w, y : y + h] = child.weight * result
        self.current = transforms.functional.resize(
            base_prediction,
            (self.crop[2], self.crop[3]),
            antialias=True,
        )
        return self.current

    def add_example(self, example, answer):
        self.data.append((example, answer))
        if len(self.incoming_edges) > 0:
            input = self.model.generate(example)
            child_results = []
            for child in self.outgoing_edges:
                child_results.append(child, child.node.add_example(input, answer))

            for child, result in child_results:
                x = child.node.crop[0]
                y = child.node.crop[1]
                w = child.node.crop[2]
                h = child.node.crop[3]
                input[:, x : x + w, y : y + h] = child.weight * result
            return input
        else:
            return None

    def clear_examples(self):
        self.data = []
        self.inputs = None
        self.outputs = None

    def train(self, epochs=101, save_interval=100):
        self.inputs = CustomDataset(
            [d[0] for d in self.data],
            transforms=transforms.Compose(
                [
                    transforms.ToTensor(),
                    lambda img: transforms.functional.crop(img, *self.crop),
                    lambda img: transforms.functional.resize(img, (32, 32)),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            ),
        )
        self.outputs = CustomDataset(
            [d[1] for d in self.data],
            transforms=transforms.Compose(
                [
                    transforms.ToTensor(),
                    lambda img: transforms.functional.crop(img, *self.crop),
                    lambda img: transforms.functional.resize(img, (32, 32)),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            ),
        )
        self.model.train(
            nn.MSELoss(),
            epochs=epochs,
            save_interval=save_interval,
            batch_size=64,
            training_data=self.inputs,
            target_data=self.outputs,
        )

    def test(self):
        raw_inputs = [render_polygon(N_SIDES) for i in range(0, 10)]

        processed_inputs = CustomDataset(
            raw_inputs,
            transforms=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            ),
        )

        outputs = []
        for test in raw_inputs:
            outputs.append(self.generate(test).cpu())
            self.clear()

        grid = make_grid(
            torch.from_numpy(np.array(outputs)), nrow=10, normalize=True, padding=1
        )
        grid = grid.permute(1, 2, 0)

        inputs = torch.cat([input for input in processed_inputs], dim=0).cpu()

        grid2 = make_grid(
            inputs.view(-1, 3, 256, 256),
            nrow=10,
            normalize=True,
            padding=1,
        )
        grid2 = grid2.permute(1, 2, 0)

        _, ax = plt.subplots(2, 1)
        ax[0].imshow(grid)
        ax[1].imshow(grid2)
        plt.axis("off")
        plt.show()


graph_network = Node("root", (0, 0, 256, 256))
graph_network.load_weights()

examples = [render_polygon(N_SIDES) for i in range(0, 100)]
for example in examples:
    graph_network.add_example(example, example)

# graph_network.train()

graph_network.test()
