# -*- coding: utf-8 -*-

import os
import logging

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np

"""
Data loaders -- handlers to load train/validation data

Supported datasets:
 - MNIST
 - CIFAR-10
 - and their versions with noisy labels

"""

def load_dataset(dataset, path, noise = 0, batch_size = 100):
  """Load the specified dataset and form mini-batches of size batch_size.

  Args
    dataset (str) -- name of the dataset
    path (str) -- path to the datasets
    batch_size (int) -- mini-batch size
    noise (float) -- percentage of noisy labels in the training set [0, 1.0]

  """

  if not os.path.exists(path):
    os.mkdir(path)
  logger = logging.getLogger('train')
  msg = 'Loading {}'.format(dataset)
  msg = msg + ', corrupt labels with probability {}'.format(noise) if noise > 0 else ''
  logger.info(msg)

  if dataset == 'mnist':
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
    # if not available locally, download mnist
    if noise > 0:
      train_set = MNISTNoisyLabels(noise=noise, root=path, train=True, transform=transform, download=False)
    else:
      train_set = datasets.MNIST(root=path, train=True, transform=transform, download=True)
    test_set = datasets.MNIST(root=path, train=False, transform=transform, download=True)
    train_loader = torch.utils.data.DataLoader(
                   dataset=train_set,
                   batch_size=batch_size,
                   shuffle=True)
    test_loader = torch.utils.data.DataLoader(
                  dataset=test_set,
                  batch_size=batch_size,
                  shuffle=False)
  elif dataset == 'cifar10':
    data_augmentation = transforms.Compose(
                        [transforms.RandomCrop(32, padding=4),
                         transforms.RandomHorizontalFlip(),
                         transforms.ToTensor(),
                         transforms.Normalize(mean = [x / 255.0 for x in [125.3, 123.0, 113.9]],
                              std = [ x / 255.0 for x in [63.0, 62.1, 66.7]])])
  
    transform_test = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean = [x / 255.0 for x in [125.3, 123.0, 113.9]],
                              std = [ x / 255.0 for x in [63.0, 62.1, 66.7]])])

    # if not available, download cifar10
    if noise > 0: # no data augmentation on random labels
      train_set = CIFAR10NoisyLabels(noise=noise, root=path, train=True, transform=transform_test, download=False)
    else:
      train_set = datasets.CIFAR10(root=path, train=True,
                                            download=True, transform=data_augmentation)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                              shuffle=True, num_workers=4)

    test_set = datasets.CIFAR10(root=path, train=False,
                                           download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                             shuffle=False, num_workers=4)
  else:
    raise ValueError("Unsupported dataset.")

  return train_loader, test_loader
  
class CIFAR10NoisyLabels(datasets.CIFAR10):
  """Load CIFAR10 with noisy labels
  
  source: https://github.com/pluskid/fitting-random-labels/blob/master/cifar10_data.py
  
  Args:
    noise (float) -- percentage of corrupted labels
    
  """
  
  def __init__(self, noise = 0, classes = 10, **kwargs):
    super(CIFAR10NoisyLabels, self).__init__(**kwargs)
    self.classes = classes
    if noise > 0:
      self.corrupt_labels(noise)
    
  def corrupt_labels(self, noise):
    logger = logging.getLogger('train')
    logger.info('Randomizing CIFAR10 labels')
    
    try:
      labels = np.array(self.targets)
    except AttributeError: # older torchvision version
      if self.train:
        labels = np.array(self.train_labels)
      else:
        labels = np.array(self.test_labels)
        
    orig_labels = labels # sanity check
    
    np.random.seed(12345) # fixed seed for reproducibility
    mask = np.random.rand(len(labels)) <= noise
    random_labels = np.random.choice(self.classes, mask.sum())
    labels[mask] = random_labels
    labels = [int(x) for int in labels]
    
    logger.debug('Sanity check -- actual ratio of corrupted labels: {}%'.format(np.sum(orig_labels != np.array(labels)) / len(labels)))
    
    try:
      self.targets = labels
    except AttributeError: # older torchvision version
      if self.train:
        self.train_labels = labels
      else:
        self.test_labels  = labels

class MNISTNoisyLabels(datasets.MNIST):
  """Load MNIST with noisy labels
  
  source: https://github.com/pluskid/fitting-random-labels/blob/master/cifar10_data.py
  
  Args:
    noise (float) -- percentage of corrupted labels
    
  """
  
  def __init__(self, noise = 0, classes = 10, **kwargs):
    super(MNISTNoisyLabels, self).__init__(**kwargs)
    self.classes = classes
    if noise > 0:
      self.corrupt_labels(noise)
    
  def corrupt_labels(self, noise):
    logger = logging.getLogger('train')
    logger.info('Randomizing MNIST labels')
    
    try:
      labels = np.array(self.targets)
    except AttributeError: # older torchvision version
      if self.train:
        labels = np.array(self.train_labels)
      else:
        labels = np.array(self.test_labels)
        
    orig_labels = labels # sanity check
    
    np.random.seed(12345) # fixed seed for reproducibility
    mask = np.random.rand(len(labels)) <= noise
    random_labels = np.random.choice(self.classes, mask.sum())
    labels[mask] = random_labels
    labels = [int(x) for x in labels]
    
    logger.debug('Sanity check -- actual ratio of corrupted labels: {}%'.format(np.sum(orig_labels != np.array(labels)) / len(labels)))
    
    try:
      self.targets = labels
    except AttributeError: # older torchvision version
      if self.train:
        self.train_labels = labels
      else:
        self.test_labels  = labels
    
