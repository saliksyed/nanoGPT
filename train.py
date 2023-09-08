#  article dependencies
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as Datasets
import torchvision.transforms.functional as tv
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from tqdm import tqdm as tqdm_regular
import seaborn as sns
from torchvision.utils import make_grid
import random


device = torch.device('mps')
print('Running on the mps')

from frame import render_polygon

N_SIDES = 4
WINDOW = 4
NUM_TRAIN = 1000
  
#  extracting training images
raw_images = [render_polygon(N_SIDES, WINDOW) for _ in range(0, NUM_TRAIN)]

training_images = []
target_images = []
for imgs in raw_images:
  example = np.concatenate(imgs[:-1], axis=1)
  answer = np.concatenate(imgs[1:], axis=1)
  training_images.append(example)
  target_images.append(answer)

#  defining dataset class
class CustomDataset(Dataset):
  def __init__(self, data, transforms=None):
    self.data = data
    self.transforms = transforms

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    image = self.data[idx]

    if self.transforms!=None:
      image = self.transforms(image)
    return image
    
    
#  creating pytorch datasets
training_data = CustomDataset(training_images, transforms=transforms.Compose([transforms.ToTensor(),
                                                                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))

target_data = CustomDataset(target_images, transforms=transforms.Compose([transforms.ToTensor(),
                                                                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))

#  defining encoder
class Encoder(nn.Module):
  def __init__(self, in_channels=3, out_channels=16, latent_dim=200, act_fn=nn.ReLU()):
    super().__init__()
    
    self.net = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1), # (32, 32)
        act_fn,
        nn.Conv2d(out_channels, out_channels, 3, padding=1), 
        act_fn,
        nn.Conv2d(out_channels, 2*out_channels, 3, padding=1, stride=2), # (16, 16)
        act_fn,
        nn.Conv2d(2*out_channels, 2*out_channels, 3, padding=1),
        act_fn,
        nn.Conv2d(2*out_channels, 4*out_channels, 3, padding=1, stride=2), # (8, 8)
        act_fn,
        nn.Conv2d(4*out_channels, 4*out_channels, 3, padding=1),
        act_fn,
        nn.Flatten(),
        nn.Linear(4*out_channels*8*8, latent_dim),
        act_fn
    )

  def forward(self, x):
    x = x.view(-1, 3, 32, 32)
    output = self.net(x)
    return output


#  defining decoder
class Decoder(nn.Module):
  def __init__(self, in_channels=3, out_channels=16, latent_dim=200, act_fn=nn.ReLU()):
    super().__init__()

    self.out_channels = out_channels

    self.linear = nn.Sequential(
        nn.Linear(latent_dim, 4*out_channels*8*8),
        act_fn
    )

    self.conv = nn.Sequential(
        nn.ConvTranspose2d(4*out_channels, 4*out_channels, 3, padding=1), # (8, 8)
        act_fn,
        nn.ConvTranspose2d(4*out_channels, 2*out_channels, 3, padding=1, 
                           stride=2, output_padding=1), # (16, 16)
        act_fn,
        nn.ConvTranspose2d(2*out_channels, 2*out_channels, 3, padding=1),
        act_fn,
        nn.ConvTranspose2d(2*out_channels, out_channels, 3, padding=1, 
                           stride=2, output_padding=1), # (32, 32)
        act_fn,
        nn.ConvTranspose2d(out_channels, out_channels, 3, padding=1),
        act_fn,
        nn.ConvTranspose2d(out_channels, in_channels, 3, padding=1)
    )

  def forward(self, x):
    output = self.linear(x)
    output = output.view(-1, 4*self.out_channels, 8, 8)
    output = self.conv(output)
    return output


#  defining autoencoder
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
  
class ConvolutionalAutoencoder():
  def __init__(self, autoencoder, checkpoint_path='ckpt.pt'):
    self.network = autoencoder
    self.checkpoint_path = checkpoint_path
    self.optimizer = torch.optim.Adam(self.network.parameters(), lr=1e-3)
  
  def load_weights(self):
    checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    self.model.network.load_state_dict(state_dict)

  def train(self, loss_function, epochs, batch_size, training_data, target_data):
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
      print(f'Epoch {epoch+1}/{epochs}')
      for i in range(0, 100):
        train = []
        target = []
        for j in range(0, batch_size):
          r = random.randint(0, NUM_TRAIN - 1)
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
        loss = loss_function(tv.crop(output.view(-1,3,32,128), 0, 96, 32, 32), tv.crop(target.view(-1, 3, 32, 128), 0, 96, 32, 32))
        # #  calculating gradients
        loss.backward()
        # #  optimizing weights
        self.optimizer.step()

      if epoch % 100 == 0:
        checkpoint = {
            'model': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        print(f"saving checkpoint")
        torch.save(checkpoint, self.checkpoint_path)
      print(f'training_loss: {round(loss.item(), 4)}')

  def test(self):
    with torch.no_grad():
      #  reconstructing test images
#  extracting training images
      raw_images = [render_polygon(5, WINDOW) for _ in range(0, 5)]

      training_images = []
      target_images = []
      for imgs in raw_images:
        example = np.concatenate(imgs[:-1], axis=1)
        answer = np.concatenate(imgs[1:], axis=1)
        training_images.append(example)
        target_images.append(answer)
      
      examples = CustomDataset(training_images, transforms=transforms.Compose([transforms.ToTensor(),
                                                                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))

      answers = CustomDataset(target_images, transforms=transforms.Compose([transforms.ToTensor(),
                                                                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))

      reconstructed_img = self.network(examples[0].to(device))
      #  sending reconstructed and images to cpu to allow for visualization
      reconstructed_img = reconstructed_img.view(3, 32, 128).cpu()
      test_image = answers[0].cpu()

      #  visualisation
      imgs = torch.stack([test_image, reconstructed_img], dim=0)
      grid = make_grid(imgs, nrow=10, normalize=True, padding=1)
      grid = grid.permute(1, 2, 0)
      plt.figure(dpi=170)
      plt.title('Original/Reconstructed')
      plt.imshow(grid)
      plt.axis('off')
      plt.show()

  def autoencode(self, x):
    return self.network(x)

  def encode(self, x):
    encoder = self.network.encoder
    return encoder(x)
  
  def decode(self, x):
    decoder = self.network.decoder
    return decoder(x)
  
#  training model

model = ConvolutionalAutoencoder(Autoencoder(Encoder(), Decoder()))
model.load_weights()
model.train(nn.MSELoss(), epochs=101, batch_size=64, training_data=training_data, target_data=target_data)

for i in range(0, 10):
  model.test()