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


class MLPBlock(torchvision.ops.MLP):
    def __init__(self, in_dim: int, mlp_dim: int, dropout: float):
        super().__init__(
            in_dim,
            [mlp_dim, in_dim],
            activation_layer=nn.GELU,
            inplace=None,
            dropout=dropout,
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        version = local_metadata.get("version", None)

        if version is None or version < 2:
            # Replacing legacy MLPBlock with MLP. See https://github.com/pytorch/vision/pull/6053
            for i in range(2):
                for type in ["weight", "bias"]:
                    old_key = f"{prefix}linear_{i+1}.{type}"
                    new_key = f"{prefix}{3*i}.{type}"
                    if old_key in state_dict:
                        state_dict[new_key] = state_dict.pop(old_key)

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


class EncoderBlock(nn.Module):
    """Transformer encoder block."""

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        self.num_heads = num_heads

        # Attention block
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=attention_dropout, batch_first=True
        )
        self.dropout = nn.Dropout(dropout)

        # MLP block
        self.ln_2 = norm_layer(hidden_dim)
        self.mlp = MLPBlock(hidden_dim, mlp_dim, dropout)

    def forward(self, input: torch.Tensor):
        torch._assert(
            input.dim() == 3,
            f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}",
        )
        x = self.ln_1(input)
        x, _ = self.self_attention(x, x, x, need_weights=False)
        x = self.dropout(x)
        x = x + input

        y = self.ln_2(x)
        y = self.mlp(y)
        return x + y


class Encoder(nn.Module):
    """Transformer Model Encoder for sequence to sequence translation."""

    def __init__(
        self,
        seq_length: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        # Note that batch_size is on the first dim because
        # we have batch_first=True in nn.MultiAttention() by default
        self.pos_embedding = nn.Parameter(
            torch.empty(1, seq_length, hidden_dim).normal_(std=0.02)
        )  # from BERT
        self.dropout = nn.Dropout(dropout)
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        for i in range(num_layers):
            layers[f"encoder_layer_{i}"] = EncoderBlock(
                num_heads,
                hidden_dim,
                mlp_dim,
                dropout,
                attention_dropout,
                norm_layer,
            )
        self.layers = nn.Sequential(layers)
        self.ln = norm_layer(hidden_dim)

    def forward(self, input: torch.Tensor):
        torch._assert(
            input.dim() == 3,
            f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}",
        )
        input = input + self.pos_embedding
        return self.ln(self.layers(self.dropout(input)))


class ViTEncoder(nn.Module):
    def __init__(
        self,
        image_size: int,
        patch_size: int,
        hidden_dim: int,
        mlp_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float = 0.0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attention_dropout: float = 0.0,
    ):
        super().__init__()
        self.image_size = image_size
        self.hidden_dim = hidden_dim
        self.patch_size = patch_size
        self.conv_proj = nn.Conv2d(
            in_channels=3,
            out_channels=hidden_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        seq_length = (image_size // patch_size) ** 2
        self.mlp_dim = mlp_dim
        self.encoder = Encoder(
            seq_length,
            num_layers,
            num_heads,
            hidden_dim,
            mlp_dim,
            dropout,
            attention_dropout,
            norm_layer,
        )

    def forward(self, input: torch.Tensor):
        x = input
        n, c, h, w = x.shape
        p = self.patch_size
        torch._assert(
            h == self.image_size,
            f"Wrong image height! Expected {self.image_size} but got {h}!",
        )
        torch._assert(
            w == self.image_size,
            f"Wrong image width! Expected {self.image_size} but got {w}!",
        )
        n_h = h // p
        n_w = w // p
        x = self.conv_proj(x)
        x = x.reshape(n, self.hidden_dim, n_h * n_w)
        x = x.permute(0, 2, 1)
        x = self.encoder(x)
        return x


class ViTDecoder(nn.Module):
    def __init__(
        self,
        image_size: int,
        patch_size: int,
        hidden_dim: int,
        mlp_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float = 0.0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attention_dropout: float = 0.0,
    ):
        super().__init__()
        self.image_size = image_size
        self.hidden_dim = hidden_dim
        seq_length = (image_size // patch_size) ** 2
        self.mlp_dim = mlp_dim
        self.encoder = Encoder(
            seq_length,
            num_layers,
            num_heads,
            hidden_dim,
            mlp_dim,
            dropout,
            attention_dropout,
            norm_layer,
        )

    def forward(self, input: torch.Tensor):
        x = input
        n, t, w = x.shape
        x = self.encoder(x)
        return x


class DepthPredictor(nn.Module):
    def __init__(
        self,
        image_size: int,
        patch_size: int,
        hidden_dim: int,
        mlp_dim: int,
        decoder_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float = 0.0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        self.image_size = image_size
        self.encoder = ViTEncoder(
            image_size,
            patch_size,
            hidden_dim,
            mlp_dim,
            num_layers,
            num_heads,
            dropout,
            norm_layer,
        )
        self.transform = nn.Linear(hidden_dim, decoder_dim)
        self.decoder = ViTDecoder(
            image_size,
            patch_size,
            decoder_dim,
            mlp_dim,
            1,
            1,
            dropout,
            norm_layer,
        )

        self.reconstruct = nn.Linear(decoder_dim, 768)

    def forward(self, input: torch.Tensor, targets: torch.Tensor = None):
        n, c, h, w = input.shape
        x = input
        torch._assert(
            h == self.image_size,
            f"Wrong image height! Expected {self.image_size} but got {h}!",
        )
        torch._assert(
            w == self.image_size,
            f"Wrong image width! Expected {self.image_size} but got {w}!",
        )
        x = self.encoder(x)
        x = self.transform(x)
        x = self.decoder(x)
        x = self.reconstruct(x)
        patches = F.unfold(targets, kernel_size=16, stride=16)
        patches = patches.permute(0, 2, 1)
        # split the targets into patches
        error = F.mse_loss(x, patches)
        return x, error


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


epochs = 100
batch_size = 32
device = "mps"
network = DepthPredictor(128, 16, 768, 3072, 768, 12, 12, 0.0)
network.train()
network.to(device)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(network.parameters(), lr=1.5e-4)


checkpoint_path = "weights/albedo_to_normal.pt"

#### Load model
checkpoint = torch.load(checkpoint_path, map_location=device)
state_dict = checkpoint["model"]
unwanted_prefix = "_orig_mod."

for k, v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
network.load_state_dict(state_dict)


# # # ### Train
for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")

    # Render all the blender images
    render_polygon()

    # compute optical flow
    end = skimage.color.rgb2gray(skimage.io.imread(f"./data/example_{i}.png"))
    start = skimage.color.rgb2gray(skimage.io.imread(f"./data/example_{i}_delta.png"))
    flow = skimage.registration.optical_flow_ilk(start, end)
    flow_color = flow_vis.flow_to_color(np.transpose(flow), convert_to_bgr=False)
    skimage.io.imsave(f"./data/example_{i}_flow.png", flow_color)

    albedo_images = []
    normal_images = []
    for i in range(0, 1000):
        albedo = f"data/example_{i}_albedo0001.png"
        target = f"data/example_{i}_normal0001.png"
        albedo_images.append(skimage.io.imread(albedo))
        normal_images.append(skimage.io.imread(target))

    curr_transforms = [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]

    inputs = CustomDataset(
        albedo_images,
        transforms=transforms.Compose(curr_transforms),
    )
    outputs = CustomDataset(
        normal_images,
        transforms=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        ),
    )

    for i in range(0, 1000):
        train = []
        target = []
        for j in range(0, batch_size):
            r = random.randint(0, len(inputs) - 1)
            train.append(inputs[r])
            target.append(outputs[r])
        images = torch.stack(train).to(device)
        target = torch.stack(target).to(device)
        # #  zeroing gradients
        optimizer.zero_grad()
        output, loss = network(images, target)
        if i % 10 == 0:
            print(loss)
        loss.backward()
        optimizer.step()

    checkpoint = {
        "model": network.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    print(f"saving checkpoint")
    torch.save(checkpoint, checkpoint_path)

# ### Test

albedo_images = []
normal_images = []
for i in range(1001, 1100):
    albedo = f"data/example_{i}_albedo0001.png"
    target = f"data/example_{i}_normal0001.png"
    albedo_images.append(skimage.io.imread(albedo))
    normal_images.append(skimage.io.imread(target))

curr_transforms = [
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

test_inputs = CustomDataset(
    albedo_images,
    transforms=transforms.Compose(curr_transforms),
)
test_outputs = CustomDataset(
    normal_images,
    transforms=transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    ),
)

idx = 76
test = torch.stack([test_inputs[idx]]).to(device)
result = torch.stack([test_outputs[idx]]).to(device)
output, loss = network(test, result)
estimate = F.fold(
    output.permute(0, 2, 1), output_size=(128, 128), kernel_size=16, stride=16
)

import matplotlib.pyplot as plt

viz = estimate.detach().to("cpu").permute(0, 2, 3, 1).numpy().reshape(128, 128, 3)
f, ax = plt.subplots(1, 2)
ax[0].imshow(viz)
ax[1].imshow(test_outputs[idx].permute(1, 2, 0))
plt.show()
