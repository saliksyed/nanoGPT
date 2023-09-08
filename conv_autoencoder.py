#  article dependencies
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import random

from config import device


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


class Encoder(nn.Module):
    def __init__(
        self, in_channels=3, out_channels=16, latent_dim=200, act_fn=nn.ReLU()
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),  # (32, 32)
            act_fn,
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            act_fn,
            nn.Conv2d(
                out_channels, 2 * out_channels, 3, padding=1, stride=2
            ),  # (16, 16)
            act_fn,
            nn.Conv2d(2 * out_channels, 2 * out_channels, 3, padding=1),
            act_fn,
            nn.Conv2d(
                2 * out_channels, 4 * out_channels, 3, padding=1, stride=2
            ),  # (8, 8)
            act_fn,
            nn.Conv2d(4 * out_channels, 4 * out_channels, 3, padding=1),
            act_fn,
            nn.Flatten(),
            nn.Linear(4 * out_channels * 8 * 8, latent_dim),
            act_fn,
        )

    def forward(self, x):
        x = x.view(-1, 3, 32, 32)
        output = self.net(x)
        return output


class Decoder(nn.Module):
    def __init__(
        self, in_channels=3, out_channels=16, latent_dim=200, act_fn=nn.ReLU()
    ):
        super().__init__()

        self.out_channels = out_channels

        self.linear = nn.Sequential(
            nn.Linear(latent_dim, 4 * out_channels * 8 * 8), act_fn
        )

        self.conv = nn.Sequential(
            nn.ConvTranspose2d(
                4 * out_channels, 4 * out_channels, 3, padding=1
            ),  # (8, 8)
            act_fn,
            nn.ConvTranspose2d(
                4 * out_channels,
                2 * out_channels,
                3,
                padding=1,
                stride=2,
                output_padding=1,
            ),  # (16, 16)
            act_fn,
            nn.ConvTranspose2d(2 * out_channels, 2 * out_channels, 3, padding=1),
            act_fn,
            nn.ConvTranspose2d(
                2 * out_channels, out_channels, 3, padding=1, stride=2, output_padding=1
            ),  # (32, 32)
            act_fn,
            nn.ConvTranspose2d(out_channels, out_channels, 3, padding=1),
            act_fn,
            nn.ConvTranspose2d(out_channels, in_channels, 3, padding=1),
        )

    def forward(self, x):
        output = self.linear(x)
        output = output.view(-1, 4 * self.out_channels, 8, 8)
        output = self.conv(output)
        return output


class Autoencoder(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.encoder.to(device)

        self.decoder = decoder
        self.decoder.to(device)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class ConvolutionalAutoencoder:
    def __init__(self, autoencoder, checkpoint_path="ckpt.pt"):
        self.network = autoencoder
        self.checkpoint_path = checkpoint_path
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=1e-3)

    def load_weights(self):
        checkpoint = torch.load(self.checkpoint_path, map_location=device)
        state_dict = checkpoint["model"]
        unwanted_prefix = "_orig_mod."
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
        self.network.load_state_dict(state_dict)

    def generate(self, image):
        image = image.to(device)
        with torch.no_grad():
            reconstructed_img = self.network(image)
            reconstructed_img = reconstructed_img.view(3, 32, 32)
            return reconstructed_img

    def train(
        self,
        loss_function,
        epochs,
        batch_size,
        training_data,
        target_data,
        save_interval=100,
    ):
        def init_weights(module):
            if isinstance(module, nn.Conv2d):
                torch.nn.init.xavier_uniform_(module.weight)
                module.bias.data.fill_(0.01)
            elif isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                module.bias.data.fill_(0.01)

        self.network.apply(init_weights)

        self.network.train()
        self.network.to(device)

        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            for i in range(0, len(training_data)):
                train = []
                target = []
                for j in range(0, batch_size):
                    r = random.randint(0, len(training_data) - 1)
                    train.append(training_data[r])
                    target.append(target_data[r])
                images = torch.stack(train).to(device)
                target = torch.stack(target).to(device)
                # #  zeroing gradients
                self.optimizer.zero_grad()
                # #  sending images to device
                # #  reconstructing images
                output = self.network(images)
                # #  computing loss
                loss = loss_function(
                    output.view(-1, 3, 32, 32), target.view(-1, 3, 32, 32)
                )
                # #  calculating gradients
                loss.backward()
                # #  optimizing weights
                self.optimizer.step()

            if epoch % save_interval == 0:
                checkpoint = {
                    "model": self.network.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                }
                print(f"saving checkpoint")
                torch.save(checkpoint, self.checkpoint_path)
            print(f"training_loss: {round(loss.item(), 4)}")

    def autoencode(self, x):
        return self.network(x)

    def encode(self, x):
        encoder = self.network.encoder
        return encoder(x)

    def decode(self, x):
        decoder = self.network.decoder
        return decoder(x)
