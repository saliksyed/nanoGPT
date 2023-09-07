#  article dependencies
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as Datasets
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

N_SIDES = 3
  
#  extracting training images
training_images = render_polygon(N_SIDES, 1000)

#  extracting validation images
validation_images = render_polygon(N_SIDES, 1000)

test_images = render_polygon(N_SIDES, 100)

#  defining dataset class
class CustomCIFAR10(Dataset):
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
training_data = CustomCIFAR10(training_images, transforms=transforms.Compose([transforms.ToTensor(),
                                                                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
validation_data = CustomCIFAR10(validation_images, transforms=transforms.Compose([transforms.ToTensor(),
                                                                                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
test_data = CustomCIFAR10(test_images, transforms=transforms.Compose([transforms.ToTensor(),
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
  def __init__(self, autoencoder):
    self.network = autoencoder
    self.optimizer = torch.optim.Adam(self.network.parameters(), lr=1e-3)

  def train(self, loss_function, epochs, batch_size, 
            training_set, validation_set, test_set):
    
    #  creating log
    log_dict = {
        'training_loss_per_batch': [],
        'validation_loss_per_batch': [],
        'visualizations': []
    } 

    #  defining weight initialization function
    def init_weights(module):
      if isinstance(module, nn.Conv2d):
        torch.nn.init.xavier_uniform_(module.weight)
        module.bias.data.fill_(0.01)
      elif isinstance(module, nn.Linear):
        torch.nn.init.xavier_uniform_(module.weight)
        module.bias.data.fill_(0.01)

    #  initializing network weights
    self.network.apply(init_weights)

    #  setting convnet to training mode
    self.network.train()
    self.network.to(device)

    for epoch in range(epochs):
      print(f'Epoch {epoch+1}/{epochs}')
      train_losses = []

      #------------
      #  TRAINING
      #------------
      print('training...')
      window_size = 4
      for _ in range(0, 100):
        train = []
        answer = []
        for j in range(0, batch_size):
          i = random.randint(window_size, int(len(training_images)/window_size) - 1)
          train.append(torch.cat([training_data[i + j] for j in range(window_size)], dim = 1).to(device))
          answer.append(torch.cat([training_data[i + j + 1] for j in range(window_size)], dim = 1).to(device))

        images = torch.stack(train)
        target = torch.stack(answer)
        # #  zeroing gradients
        self.optimizer.zero_grad()
        # #  sending images to device
        # #  reconstructing images
        output = self.network(images)
        # #  computing loss
        loss = loss_function(output.view(-1,3,128,32), target.view(-1, 3, 128, 32))
        # #  calculating gradients
        loss.backward()
        # #  optimizing weights
        self.optimizer.step()

        #--------------
        # LOGGING
        #--------------
        log_dict['training_loss_per_batch'].append(loss.item())

      #--------------
      # VALIDATION
      #--------------
      print('validating...')
      # for i in range(window_size, int(len(val_images)/window_size - 1):
      #   with torch.no_grad():
      #     #  sending validation images to device
      #     val_images = torch.cat([validation_data[i + j] for j in range(window_size)], dim = 1).to(device)
      #     #  reconstructing images
      #     output = self.network(val_images)
      #     #  computing validation loss
      #     val_loss = loss_function(output, val_images.view(-1, 3, 32, 32))

      #   #--------------
      #   # LOGGING
      #   #--------------
        # log_dict['validation_loss_per_batch'].append(val_loss.item())


      #--------------
      # VISUALISATION
      #--------------
      print(f'training_loss: {round(loss.item(), 4)}')
      if epoch % 1000 == 0:
        print("Visualizing epoch", epoch)
        window_size = 4
        for i in range(window_size, int(len(test_images)/window_size) - 1):
          test_image = torch.cat([test_data[i + j] for j in range(window_size)], dim = 1)
          test_image = test_image.to(device)

          with torch.no_grad():
            #  reconstructing test images
            reconstructed_img = self.network(test_image)
          #  sending reconstructed and images to cpu to allow for visualization
          reconstructed_img = reconstructed_img.view(3, 128, 32).cpu()
          test_image = test_image.cpu()

          #  visualisation
          imgs = torch.stack([test_image, reconstructed_img], dim=0)
          grid = make_grid(imgs, nrow=10, normalize=True, padding=1)
          grid = grid.permute(1, 2, 0)
          plt.figure(dpi=170)
          plt.title('Original/Reconstructed')
          plt.imshow(grid)
          log_dict['visualizations'].append(grid)
          plt.axis('off')
          plt.show()
      
    return log_dict

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

log_dict = model.train(nn.MSELoss(), epochs=1000, batch_size=64, 
                       training_set=training_data, validation_set=validation_data,
                       test_set=test_data)