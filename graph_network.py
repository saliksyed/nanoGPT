import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import random
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
        self.id = id
        self.model = ConvolutionalAutoencoder(
            Autoencoder(Encoder(), Decoder()),
            checkpoint_path=os.path.join(WEIGHTS_DIR, id + ".pt"),
        )
        self.crop = crop
        self.incoming_edges = []
        self.outgoing_edges = []
        self.data = []

    def load_weights(self):
        try:
            self.model.load_weights()
        except:
            pass

    def add_child(self, node, init_weight=0.5):
        self.outgoing_edges.append(Edge(init_weight, node))
        node.incoming_edges.append(Edge(init_weight, self))

    def generate(self, example):
        curr_transforms = []

        if not torch.is_tensor(example):
            curr_transforms.append(transforms.ToTensor())

        curr_transforms += [
            lambda img: transforms.functional.crop(img, *self.crop),
            lambda img: transforms.functional.resize(
                img,
                (32, 32),
                antialias=True,
            ),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
        # crop and normalize the example
        d = CustomDataset(
            [example],
            transforms=transforms.Compose(curr_transforms),
        )
        base_prediction = transforms.functional.resize(
            self.model.generate(d[0]).cpu(),
            (self.crop[2], self.crop[3]),
            antialias=True,
        )

        child_results = []
        for child in self.outgoing_edges:
            child_results.append((child, child.node.generate(example)))
        # resize the output
        for child, result in child_results:
            x = child.node.crop[0]
            y = child.node.crop[1]
            w = child.node.crop[2]
            h = child.node.crop[3]
            # base_prediction[:, x : x + w, y : y + h] = result
        return base_prediction

    def add_example(self, example, answer):
        self.data.append((example, answer))
        if len(self.outgoing_edges) > 0:
            input = self.generate(example).view((self.crop[2], self.crop[3], 3)).numpy()
            child_results = []
            for child in self.outgoing_edges:
                child_results.append((child, child.node.add_example(input, answer)))

    def clear_examples(self):
        self.data = []
        self.inputs = None
        self.outputs = None

    def train(self, epochs=101):
        curr_transforms = [
            transforms.ToTensor(),
            lambda img: transforms.functional.crop(img, *self.crop),
            lambda img: transforms.functional.resize(
                img,
                (32, 32),
                antialias=True,
            ),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
        self.inputs = CustomDataset(
            [d[0] for d in self.data],
            transforms=transforms.Compose(curr_transforms),
        )
        self.outputs = CustomDataset(
            [d[1] for d in self.data],
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
        self.model.train(
            nn.MSELoss(),
            epochs=epochs,
            batch_size=64,
            training_data=self.inputs,
            target_data=self.outputs,
        )

    def test(self):
        cropped_inputs = [
            self.data[i][1][
                self.crop[0] : self.crop[0] + self.crop[2],
                self.crop[1] : self.crop[1] + self.crop[3],
            ]
            for i in range(0, 10)
        ]

        model_inputs = [self.data[i][0] for i in range(0, 10)]

        curr_transforms = []

        if not torch.is_tensor(cropped_inputs[0]):
            curr_transforms.append(transforms.ToTensor())

        curr_transforms += [
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]

        processed_inputs = CustomDataset(
            cropped_inputs,
            transforms=transforms.Compose(curr_transforms),
        )

        outputs = []
        for test in model_inputs:
            outputs.append(self.generate(test).cpu())

        grid = make_grid(
            torch.from_numpy(np.array(outputs)), nrow=10, normalize=True, padding=1
        )
        grid = grid.permute(1, 2, 0)

        inputs = torch.stack([input for input in processed_inputs], dim=0).cpu()

        grid2 = make_grid(
            inputs.view(-1, 3, self.crop[2], self.crop[3]),
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

    def subdivide(self, depth, so_far=1):
        if depth == 0:
            return
        x = self.crop[0]
        y = self.crop[1]
        w = self.crop[2]
        h = self.crop[3]
        nw = Node(self.id + "-nw-" + str(so_far), (0, 0, w // 2, h // 2))
        ne = Node(self.id + "-ne-" + str(so_far), (0, h // 2, w // 2, h // 2))
        sw = Node(self.id + "-sw-" + str(so_far), (w // 2, 0, w // 2, h // 2))
        se = Node(self.id + "-se-" + str(so_far), (w // 2, h // 2, w // 2, h // 2))
        self.add_child(nw)
        self.add_child(ne)
        self.add_child(sw)
        self.add_child(se)
        depth -= 1
        nw.subdivide(depth, so_far + 1)
        ne.subdivide(depth, so_far + 1)
        sw.subdivide(depth, so_far + 1)
        se.subdivide(depth, so_far + 1)

    def nodes(self):
        ret = [self]
        all_descendents = [edge.node.nodes() for edge in self.outgoing_edges]
        for node_descendents in all_descendents:
            ret += node_descendents
        return ret


g = Node("root", (0, 0, 256, 256))
g.subdivide(1)
nodes = g.nodes()
for node in nodes:
    node.load_weights()

# examples = [render_polygon(N_SIDES) for i in range(0, 1000)]
# for example in examples:
#     g.add_example(example, example)

# for node in nodes:
#     node.train(epochs=1)

for node in nodes:
    node.clear_examples()

examples = [render_polygon(N_SIDES) for i in range(0, 10)]
for example in examples:
    g.add_example(example, example)
g.test()
