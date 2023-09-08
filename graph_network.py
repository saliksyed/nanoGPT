import os
import uuid

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as Datasets
import torchvision.transforms.functional as tv
from torch.utils.data import Dataset, DataLoader

from train import CustomDataset, ConvolutionalAutoencoder, Autoencoder, Encoder, Decoder

WEIGHTS_DIR = 'weights'

os.mkdir(WEIGHTS_DIR, exist_ok=True)

class Edge:
    def __init__(self, weight, node):
        self.weight = weight
        self.node = node

class Node:
    def __init__(self, id, crop, device):
        self.model = ConvolutionalAutoencoder(Autoencoder(Encoder(), Decoder()), checkpoint_path=os.path.join(WEIGHTS_DIR, id + '.pt'))
        self.device = device
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
        if self.current:
            return self.current
        # crop and normalize the example
        d = CustomDataset([example], transforms=transforms.Compose([transforms.ToTensor(), lambda img : transforms.functional.crop(img, *self.crop), lambda img: transforms.functional.resize(img, (32, 32)) ,transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
        # make prediction
        base_prediction = self.model.generate(d[0])
        child_results = []
        for child in self.incoming_edges:
            child_results.append(child, child.node.generate(example))
        # resize the output
        for child, result in child_results:
            x = child.node.crop[0]
            y = child.node.crop[1]
            w = child.node.crop[2]
            h = child.node.crop[3]
            base_prediction[:, x:x+w, y:y+h] = child.weight * result
        self.current = transforms.functional.resize(base_prediction, (self.crop[2], self.crop[3]))
        return self.current

    def add_example(self, example, answer):
        self.data.append(example, answer)
        input = self.model.generate(example)
        
        child_results = []

        for child in self.outgoing_edges:
            child_results.append(child, child.node.add_example(input, answer))
        
        for child, result in child_results:
            x = child.node.crop[0]
            y = child.node.crop[1]
            w = child.node.crop[2]
            h = child.node.crop[3]
            input[:, x:x+w, y:y+h] = child.weight * result
        return input

    def clear_examples(self):
        self.data = []
        self.inputs = None
        self.outputs = None

    def train(self):
        self.inputs = CustomDataset([d[0] for d in self.data], transforms=transforms.Compose([transforms.ToTensor(), lambda img : transforms.functional.crop(img, *self.crop), lambda img: transforms.functional.resize(img, (32, 32)) ,transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
        self.outputs = CustomDataset([d[1] for d in self.data], transforms=transforms.Compose([transforms.ToTensor(), lambda img : transforms.functional.crop(img, *self.crop), lambda img: transforms.functional.resize(img, (32, 32)) ,transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
        self.model.train(nn.MSELoss(), epochs=101, batch_size=64, training_data=self.inputs, target_data=self.outputs)