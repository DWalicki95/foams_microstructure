# -*- coding: utf-8 -*-
"""helper_functions.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1RZbqvFs40-pC5PYBPVi8yo4K5_SWpOQu
"""

from google.colab import drive
drive.mount('/content/drive')

import os
import collections
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision import datasets
from torch.utils.data import Dataset
import torchvision
from tqdm import tqdm
from torch import nn
import torchvision.models as models
from torchvision.models import resnet50, ResNet50_Weights
from torch.optim.lr_scheduler import StepLR

def count_files_in_drive(folder_file_count: dict, zoom: int, DRIVE_PATH: str):
  '''Function that returns number of files in drive path'''

  for root, dirs, files in os.walk(DRIVE_PATH):
    count = 0
    folder_name = os.path.basename(root)

    for file in files:
      filename = folder_name+'_'+str(zoom)

      if file.startswith(filename) & file.endswith('.jpg'):
          count += 1

    if folder_name.startswith('A'):
      folder_file_count[filename] = count

  return folder_file_count

def cut_img(image, left=0, top=0, right=1280, bottom=960):
  '''
  Cutting image. Takes input:
    * left = 'x' coordination
    * right = left + img width
    * top = 'y' coordination
    * bottom = top + height
  '''

  return image.crop((left, top, right, bottom))

def mask_img(image, mask_size=(2, 2), masked_places=1, seed=42, mask_return=False):

  '''
  Mask random part of image.
  '''

  height, width = image.size
  masked_image = image.copy()

  for place in range(masked_places):

    random.seed(seed)
    top = np.random.randint(0, height - mask_size[0])
    left = np.random.randint(0, width - mask_size[1])
    bottom = top + mask_size[0]
    right = left + mask_size[1]

    # image.crop((left, top, right, bottom))
    draw = ImageDraw.Draw(masked_image)
    draw.rectangle([left, top, right, bottom], fill=0)

  if mask_return:
    return masked_image, (left, top, right, bottom)

  else:
    return masked_image

def create_dataset(PATH: str, zoom: int, data: pd.DataFrame, output_property: str):
  '''This function creates dataset - each sample with different magnifications is
      connected with chosen property'''

  image_paths = []
  properties = []
  samples = []
  for root, dirs, files in os.walk(PATH):
    folder_name = os.path.basename(root)
    for file in files:
      filename = folder_name+'_'+str(zoom)
      if file.startswith(filename) & file.endswith('.jpg'):
        image_path = os.path.join(root, file)
        folder_name = os.path.basename(os.path.dirname(image_path))
        try:
          property_values = data.loc[data['sample_name'] == folder_name, output_property].values[0]
          properties.append(property_values)
          image_paths.append(image_path)
          samples.append(folder_name)
        except IndexError:
          continue

  df = pd.DataFrame({
      'image_path': image_paths,
      output_property: properties,
      'sample': samples
  })

  return df

def print_random_image(dataset: pd.DataFrame, return_rand_img=False):
  image_path_list = list(dataset['image_path'])

  random_number = random.randint(0, dataset.shape[0])
  random_image_path, random_property = dataset.loc[random_number][0], dataset.loc[random_number][1]
  random_property_name = dataset.columns[1]
  random_sample_name = dataset.loc[random_number][2]

  img = Image.open(random_image_path)

  print(f'Random image path: {random_image_path}')
  print(f'Sample: {random_sample_name}')
  print(f'Property({random_property_name}): {random_property}')
  print(f'Image height: {img.height}')
  print(f'Image width: {img.width}')
  img.show()

  if return_rand_img:
    return img

def count_init_transform_shape(height: int, width: int):

  ratio = height/width

  if height > width:
    new_height = 224
    new_width = new_height * 1/ratio

  else:
    new_width = 224
    new_height = new_width *  ratio

  return int(new_height), int(new_width)

def split_dataset(dataset: pd.DataFrame, TEST_SIZE=0.2,  RANDOM_STATE=42):
  X_train, X_test, y_train, y_test = train_test_split(dataset.iloc[:, 0], dataset.iloc[:, 1], test_size=TEST_SIZE, stratify=dataset.iloc[:, 2], random_state=RANDOM_STATE)
  return X_train.reset_index(drop=True),  X_test.reset_index(drop=True),  y_train.reset_index(drop=True),  y_train.reset_index(drop=True)

def display_random_images(dataset: torch.utils.data.dataset.Dataset,
                          n: int = 5,
                          display_shape: bool = True,
                          seed: int = None,
                          grayscale: bool = False):
  if n > 10:
    display_shape = False
    n = 10
    print(f'n shouldt be larger than 10')

  if seed:
    random.seed(seed)

  random_samples_idx = random.sample(range(len(dataset)), k=n)

  plt.figure(figsize=(16, 12))

  for i, targ_sample in enumerate(random_samples_idx):
    targ_image, targ_label = dataset[targ_sample][0], dataset[targ_sample][1]

    targ_image_adjust = targ_image.permute(1, 2, 0)

    plt.subplot(1, n, i+1)
    if grayscale:
      plt.imshow(targ_image_adjust, cmap='gray')
    else:
      plt.imshow(targ_image_adjust)
    plt.axis('off')
    title = f'Property value: {targ_label}'
    if display_shape:
      title = title + f'\nshape: {list(targ_image_adjust.shape)}'
    plt.title(title)

class CustomImageDataset(torch.utils.data.Dataset):

  def __init__(self, X_train, y_train, transform=None):
    self.img_label = y_train
    self.img_path = X_train
    self.transform = transform

  def __len__(self):
    return len(self.img_path)

  def __getitem__(self, idx):
    img_path = self.img_path.iloc[idx]
    image = Image.open(img_path)
    if self.transform:
      image = self.transform(image)
    label = self.img_label.iloc[idx]
    return image, label

def train_step(model: torch.nn.Module,
               train_dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device):
  train_loss, train_metric = 0, 0
  train_loss_list = []
  step_size = 1 #scheduler lr parameter
  gamma = 0.90 #scheduler lr parameter
  scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma) #added learning_rate scheduler

  model.to(device)
  for batch, (X, y) in enumerate(train_dataloader):
    X, y = X.float().to(device), y.float().to(device)

    #Forward pass
    y_pred = model(X)
    # y_pred = y_pred.view(y.shape)

    #Calculate loss
    # print(f'Wymiary y_pred: {y_pred.shape}')
    # print(f'Wymiary y: {y.shape}')
    loss = loss_fn(y_pred, y) #było y.squeeze()
    train_loss += loss.item()

    #Optimizer zero grad
    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

  train_loss /= len(train_dataloader)

  scheduler.step() #learning rate upgrade
  current_lr = scheduler.get_last_lr()[0] #get actual learning_rate value

  print(f'Train loss: {train_loss:.5f} | Learning rate: {current_lr}')

  return train_loss

def test_step(model: torch.nn.Module,
              test_dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device: torch.device):
  test_loss, test_metric = 0, 0
  test_loss_list = []

  model.to(device)
  model.eval()
  with torch.inference_mode():
    for X, y in test_dataloader:
      X, y = X.float().to(device), y.float().to(device)

      test_pred = model(X)
      # test_pred = test_pred.view(y.shape)

      # print(f'Wymiary y_pred: {test_pred.shape}')
      # print(f'Wymiary y: {y.shape}')
      test_loss += loss_fn(test_pred, y).item() #było i.squeeze()

    test_loss /= len(test_dataloader)

    print(f'Test loss: {test_loss:.5f}')

    return test_loss

def train_and_test(model: torch.nn.Module,
               train_dataloader: torch.utils.data.DataLoader,
               val_dataloader: torch.utils.data.DataLoader,
               test_dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: str,
               device: torch.device,
               save_model_name: str,
               learning_rate: float = 0.1,
               epochs: int = 5,
               patience: int = 3):

  '''Args:
      optimizer: str - Possible SGD or Adam
  '''


  if optimizer == 'SGD':
    optimizer = torch.optim.SGD(params=model.parameters(),
                              lr=learning_rate)
  elif optimizer == 'Adam':
    optimizer = torch.optim.Adam(params=model.parameters(),
                              lr=learning_rate)

  scheduler = StepLR(optimizer, step_size=1, gamma=0.95)

  train_loss_list, val_loss_list, test_loss_list = [], [], []

  #early stopping
  n_epochs_stop = patience
  min_val_loss = np.inf
  epochs_no_imporve = 0

  for epoch in tqdm(range(epochs)):
    train_loss = train_step(model = model,
               train_dataloader = train_dataloader,
               loss_fn = loss_fn,
               optimizer = optimizer,
               device = device)
    train_loss_list.append(train_loss)

    val_loss = test_step(model=model,
                         test_dataloader=val_dataloader,
                         loss_fn=loss_fn,
                         device=device)
    val_loss_list.append(val_loss)


    test_loss = test_step(model = model,
              test_dataloader = test_dataloader,
              loss_fn = loss_fn,
              device = device)
    test_loss_list.append(test_loss)

    epoch_counter = epoch+1

    if val_loss < min_val_loss:
      torch.save(model.state_dict(), save_model_name)
      epochs_no_improve = 0
      min_val_loss = val_loss

    else:
      epochs_no_improve += 1

      if epochs_no_improve == n_epochs_stop:
        print('Early stopping!')
        model.load_state_dict(torch.load(save_model_name))
        break
  return train_loss_list, test_loss_list, epoch_counter

def learning_curve(epoch: int,
                   train_loss: float,
                   test_loss: float):

  '''
  Function prints learning curves.
  '''

  plt.figure(figsize=(15, 7))


  plt.subplot(1, 2, 1)
  plt.plot(range(epoch), train_loss, label='train_loss')
  plt.plot(range(epoch), test_loss, label='test_loss')
  plt.title('Loss - original shape')
  plt.xlabel('Epochs')
  plt.legend()

  plt.subplot(1, 2, 2)
  plt.plot(range(2, epoch), train_loss[2:], label='train_loss')
  plt.plot(range(2, epoch), test_loss[2:], label='test_loss')
  plt.title(f'Loss - from epoch 2')
  plt.xlabel('Epochs')
  plt.legend()

class CustomImageDataset_OwnNN(torch.utils.data.Dataset):

  def __init__(self, X_train, y_train, transform=None):
    self.img_label = y_train
    self.img_path = X_train
    self.transform = transform

  def __len__(self):
    return len(self.img_path)

  def __getitem__(self, idx):
    img_path = self.img_path.iloc[idx]
    image = Image.open(img_path).convert('L') #grayscale
    if self.transform:
      image = self.transform(image)
    label = torch.tensor(self.img_label.iloc[idx], dtype=torch.long)
    return image, label

class ModCustomMaskedImageDataset(CustomImageDataset):
  def __init__(self, paths, transform=None):
    self.img_paths = paths
    self.transform = transform

  def __len__(self):
    return len(self.img_paths)

  def __getitem__(self, idx):
    image, label = super().__getitem__(idx)
    return image, sample

def count_images_mean_std(dataloader: torch.utils.data.DataLoader):
  '''
  This function returns means and standard deviation of images as array of pixels.
  '''
  mean = 0
  std = 0

  n_images = 0 #number of images

  for images, _ in dataloader:

    images = images.squeeze(1) #removing the dimension of channels

    images = images.view(images.size(0), -1) #reshape the images into a flat array

    mean += images.mean(1).sum().item()
    std += images.std(1).sum().item()

    n_images += images.size(0)

  mean /= n_images
  std /= n_images

  print(f'Mean: {mean} | Std: {std}' )

  return mean, std

def show_transformed_images(image_paths, transform, n=3, seed=42):
  random.seed(seed)
  random_image_paths = random.sample(image_paths, k=n)
  for image_path in random_image_paths:
    with Image.open(image_path).convert('L') as f:
      width, height = f.size

      # cut_point = int(height*)

      fig, ax = plt.subplots(1, 2)

      ax[0].imshow(f, cmap='gray')
      ax[0].set_title(f'Original \nSize: {f.size}')
      ax[0].axis('off')

      transformed_image = transform(f).squeeze(0)
      transformed_image_array = np.array(transformed_image)

      ax[1].imshow(transformed_image_array, cmap='gray')
      ax[1].set_title(f'Transformed \nSize: {transformed_image_array.shape}')
      ax[1].axis('off')

def show_predicted_mask(preds, sample, index: int = 0):
  '''
  Shows original image vs predicted one from ContextPredictor model.
  :param: index: index of an image in batch. Choose to see different images

  '''

  plt.figure(figsize=(12, 6))
  plt.subplot(1, 2, 1)
  plt.imshow(sample[index].permute(1, 2, 0).numpy(), cmap='gray')
  plt.title('Original image')

  plt.subplot(1, 2, 2)
  plt.imshow(preds[index].squeeze().cpu().numpy(), cmap='gray')
  plt.title('Predicted mask')

  plt.show()


def plot_train_vs_pred(model: torch.nn.Module,
                       test_dataloader: torch.utils.data.DataLoader,
                       device: str):
  model.eval()
  model.to(device)
  true_vals, pred_vals = [], []

  with torch.inference_mode():
    for X, y in test_dataloader:
      X, y = X.to(device), y.to(device)
      pred = model(X)

      true_vals.append(y.detach().cpu().numpy())
      pred_vals.append(pred.detach().cpu().numpy())

  true_vals = np.concatenate(true_vals, axis=0)
  pred_vals = np.concatenate(pred_vals, axis=0)

  plt.figure(figsize=(10, 8))

  plt.scatter(true_vals, pred_vals, alpha=0.5)
  plt.xlabel('True values')
  plt.ylabel('Predicted Values')
  plt.title('True vs predicted values')

  #perfect fit
  min_val = min(min(true_vals), min(pred_vals))
  max_val = max(max(true_vals), max(pred_vals))
  plt.plot([min_val, max_val], [min_val, max_val], color='red')

  plt.grid()
  plt.show()


def learning_curve(epoch: int,
                   train_loss: float,
                   test_loss: float):

  plt.figure(figsize=(15, 7))


  plt.subplot(1, 2, 1)
  plt.plot(range(epoch), train_loss, label='train_loss')
  plt.plot(range(epoch), test_loss, label='test_loss')
  plt.title('Loss - original shape')
  plt.xlabel('Epochs')
  plt.legend()

  plt.subplot(1, 2, 2)
  plt.plot(range(2, epoch), train_loss[2:], label='train_loss')
  plt.plot(range(2, epoch), test_loss[2:], label='test_loss')
  plt.title(f'Loss - from epoch 2')
  plt.xlabel('Epochs')
  plt.legend()