# -*- coding: utf-8 -*-

""" Model definitions
    
    Family of implemented models:
      - VGG7 (no BN)
      - VGG9 (no BN)
      - VGG11exp (no BN)
      - VGG11 (no BN)
      - VGG13 (no BN)
      - VGG16 (no BN)
      - VGG19 (no BN)
      - LeNet5
      - LeNet8
"""
import torch
import torch.nn as nn

import math

__teachers__ = ['VGG7', 'VGG9', 'VGG11exp', 'VGG11', 'VGG13', 'VGG16', 'VGG19', 'LeNet5', 'LeNet8', 'LeNet9']
__students__ = ['Student_VGG7', 'Student_VGG9', 'Student_VGG11exp', 'Student_VGG11', 'Student_VGG13', 'Student_VGG16', 'Student_VGG19', 'Student_LeNet5', 'Student_LeNet8', 'Student_LeNet9']
__all__ = ['VGG7', 'VGG9', 'VGG11exp', 'VGG11', 'VGG13', 'VGG16', 'VGG19', 'LeNet5', 'LeNet8', 'LeNet9', 'Student_VGG7', 'Student_VGG9', 'Student_VGG11exp', 'Student_VGG11', 'Student_VGG13', 'Student_VGG16', 'Student_VGG19', 'Student_LeNet5', 'Student_LeNet8', 'Student_LeNet9']

class LeNet(nn.Module):
  def __init__(self, features, num_classes=10, in_linear=784, init_weights='x_gaussian'):
    super(LeNet, self).__init__()
    self.features = features
    self.classifier = nn.Sequential(
      nn.Linear(in_linear, 120),
      nn.ReLU(True),
      nn.Dropout(),
      nn.Linear(120, 84),
      nn.ReLU(True),
      nn.Dropout(),
      nn.Linear(84, num_classes),
    )
    self.__name__ = 'LeNet'
    if init_weights is not None:
      self._initialize_weights(init_weights)
      
  def forward(self, x):
    x = self.features(x)
    x = x.view(x.size(0), -1)
    x = self.classifier(x)
    return x
   
  def _initialize_weights(self, init):
    gain = nn.init.calculate_gain('relu')
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        if init == 'x_gaussian':
          nn.init.xavier_normal_(m.weight, gain=gain)
        elif init == 'x_uniform':
          nn.init.xavier_uniform_(m.weight, gain=gain)
        elif init == 'h_uniform':
          nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        elif init == 'h_gaussian':
          nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        else:
          n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
          m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
          m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
          if init == 'x_gaussian':
            nn.init.xavier_normal_(m.weight, gain=gain)
          elif init == 'x_uniform':
            nn.init.xavier_uniform_(m.weight, gain=gain)
          elif init == 'h_uniform':
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
          elif init == 'h_gaussian':
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
          else:
            m.weight.data.normal_(0, 0.01)
          m.bias.data.zero_()

class VGG(nn.Module):
  
  def __init__(self, features, num_classes=10, in_linear=512, init_weights='x_gaussian'):
    super(VGG, self).__init__()
    self.features = features
    self.classifier = nn.Sequential(
      nn.Linear(in_linear, 512),
      nn.ReLU(True),
      nn.Dropout(),
      nn.Linear(512, 512),
      nn.ReLU(True),
      nn.Dropout(),
      nn.Linear(512, num_classes),
    )
    self.__name__ = 'VGG'
    if init_weights is not None:
      self._initialize_weights(init_weights)
      
  def forward(self, x):
    x = self.features(x)
    x = x.view(x.size(0), -1)
    x = self.classifier(x)
    return x
    
  def _initialize_weights(self, init):
    gain = nn.init.calculate_gain('relu')
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        if init == 'x_gaussian':
          nn.init.xavier_normal_(m.weight, gain=gain)
        elif init == 'x_uniform':
          nn.init.xavier_uniform_(m.weight, gain=gain)
        elif init == 'h_uniform':
          nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        elif init == 'h_gaussian':
          nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        else:
          n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
          m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
          m.bias.data.zero_()
      elif isinstance(m, nn.Linear):
        if init == 'x_gaussian':
          nn.init.xavier_normal_(m.weight, gain=gain)
        elif init == 'x_uniform':
          nn.init.xavier_uniform_(m.weight, gain=gain)
        elif init == 'h_uniform':
          nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        elif init == 'h_gaussian':
          nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        else:
          m.weight.data.normal_(0, 0.01)
        m.bias.data.zero_()
        
class LeNetStudent(nn.Module):
  
  def __init__(self, features, num_classes=10, in_linear=784, init_weights='x_gaussian'):
    super(LeNetStudent, self).__init__()
    self.features = features
    self.classifier = nn.Sequential(
      nn.Linear(in_linear, 120),
      nn.ReLU(True),
      nn.Dropout(),
      nn.Linear(120, num_classes),
    )
    self.__name__ = 'Student'
    if init_weights is not None:
      self._initialize_weights(init_weights)
      
  def forward(self, x):
    x = self.features(x)
    x = x.view(x.size(0), -1)
    x = self.classifier(x)
    return x
    
  def _initialize_weights(self, init):
    gain = nn.init.calculate_gain('relu')
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        if init == 'x_gaussian':
          nn.init.xavier_normal_(m.weight, gain=gain)
        elif init == 'x_uniform':
          nn.init.xavier_uniform_(m.weight, gain=gain)
        elif init == 'h_uniform':
          nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        elif init == 'h_gaussian':
          nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        else:
          n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
          m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
          m.bias.data.zero_()
      elif isinstance(m, nn.Linear):
        if init == 'x_gaussian':
          nn.init.xavier_normal_(m.weight, gain=gain)
        elif init == 'x_uniform':
          nn.init.xavier_uniform_(m.weight, gain=gain)
        elif init == 'h_uniform':
          nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        elif init == 'h_gaussian':
          nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        else:
          m.weight.data.normal_(0, 0.01)
        m.bias.data.zero_()
      elif isinstance(m, Bias):
        if m.bias is not None:
          m.bias.data.zero_()     

class VGGStudent(nn.Module):
  
  def __init__(self, features, num_classes=10, in_linear=512, init_weights='x_gaussian'):
    super(VGGStudent, self).__init__()
    self.features = features
    self.classifier = nn.Sequential(
      nn.Linear(in_linear, 512),
      nn.ReLU(True),
      nn.Dropout(),
      nn.Linear(512, 512),
      nn.ReLU(True),
      nn.Dropout(),
      nn.Linear(512, num_classes),
    )
    self.__name__ = 'Student'
    if init_weights is not None:
      self._initialize_weights(init_weights)
      
  def forward(self, x):
    x = self.features(x)
    x = x.view(x.size(0), -1)
    x = self.classifier(x)
    return x
    
  def _initialize_weights(self, init):
    gain = nn.init.calculate_gain('relu')
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        if init == 'x_gaussian':
          nn.init.xavier_normal_(m.weight, gain=gain)
        elif init == 'x_uniform':
          nn.init.xavier_uniform_(m.weight, gain=gain)
        elif init == 'h_uniform':
          nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        elif init == 'h_gaussian':
          nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        else:
          n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
          m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
          m.bias.data.zero_()
      elif isinstance(m, nn.Linear):
        if init == 'x_gaussian':
          nn.init.xavier_normal_(m.weight, gain=gain)
        elif init == 'x_uniform':
          nn.init.xavier_uniform_(m.weight, gain=gain)
        elif init == 'h_uniform':
          nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        elif init == 'h_gaussian':
          nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        else:
          m.weight.data.normal_(0, 0.01)
        m.bias.data.zero_()
      elif isinstance(m, Bias):
        if m.bias is not None:
          m.bias.data.zero_()
        
def make_layers(cfg, in_channels = 3):
  layers = []
  for v in cfg:
    if v == 'M':
      layers += [nn.MaxPool2d(kernel_size = 2, stride = 2)]
    else:
      conv2d = nn.Conv2d(in_channels, v, kernel_size = 3, padding = 1)
      layers += [conv2d, nn.ReLU(inplace = True)]
      in_channels = v
  return nn.Sequential(*layers)
  
def adaptive_linear(cfg, input_size):
  """Compute the input dimensionality of the
     first fully connected layer of the classifier
     
     input_size is a tuple (width, height, channels)
  """
  width = input_size[0]
  height = input_size[1]
  channels = input_size[2]
  
  for layer in cfg:
    if layer == 'M':
      width = width // 2
      height = height // 2
    else:
      channels = layer
  return int(width * height * channels)
  
cfg = {
  'VGG' : {
            'A'  : [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'B'  : [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'D'  : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
            'E'  : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
            'B1' : [64, 64, 'M', 128, 128, 'M'],
            'B2' : [64, 64, 'M', 128, 128, 'M', 256, 256, 'M'],
            'B3' : [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M'],
          },
  'LeNet' : {
            '5' : [6, 'M', 16, 'M'], 
            '8' : [6, 6, 'M', 16, 16, 'M'],
            '9' : [6, 6, 'M', 16, 16, 'M', 64, 64, 'M']
            }
}

def vgg7(input_size, init, **kwargs):
  """VGG 7-Layer model (experimental)
  
  Args:
    input_size: the input shape (width, height, channels)
  """
  in_linear = adaptive_linear(cfg['VGG']['B1'], input_size)
  model = VGG(make_layers(cfg['VGG']['B1'], input_size[2]), in_linear=in_linear, init_weights=init, **kwargs)
  model.__name__ = 'VGG7'
  return model
  
def vgg9(input_size, init, **kwargs):
  """VGG 9-Layer model (experimental)
  
  Args:
    input_size: the input shape (width, height, channels)
  """
  in_linear = adaptive_linear(cfg['VGG']['B2'], input_size)
  model = VGG(make_layers(cfg['VGG']['B2'], input_size[2]), in_linear=in_linear, init_weights=init, **kwargs)
  model.__name__ = 'VGG9'
  return model
  
def vgg11exp(input_size, init, **kwargs):
  """VGG 11-Layer model (experimental)
  
  Args:
    input_size: the input shape (width, height, channels)
  """
  in_linear = adaptive_linear(cfg['VGG']['B3'], input_size)
  model = VGG(make_layers(cfg['VGG']['B3'], input_size[2]), in_linear=in_linear, init_weights=init, **kwargs)
  model.__name__ = 'VGG11exp'
  return model

def vgg11(input_size, init, **kwargs):
  """VGG 11-Layer model (configuration "A")
  
  Args:
    input_size: the input shape (width, height, channels)
  """
  in_linear = adaptive_linear(cfg['VGG']['A'], input_size)
  model = VGG(make_layers(cfg['VGG']['A'], input_size[2]), in_linear=in_linear, init_weights=init, **kwargs)
  model.__name__ = 'VGG11'
  return model
  
def vgg13(input_size, init, **kwargs):
  """VGG 13-Layer model (configuration "B")
  
  Args:
    input_size: the input shape (width, height, channels)
  """
  in_linear = adaptive_linear(cfg['VGG']['B'], input_size)
  model = VGG(make_layers(cfg['VGG']['B'], input_size[2]), in_linear=in_linear, init_weights=init, **kwargs)
  model.__name__ = 'VGG13'
  return model
  
def vgg16(input_size, init, **kwargs):
  """VGG 16-Layer model (configuration "D")
  
  Args:
    input_size: the input shape (width, height, channels)
  """
  in_linear = adaptive_linear(cfg['VGG']['D'], input_size)
  model = VGG(make_layers(cfg['VGG']['D'], input_size[2]), in_linear=in_linear, init_weights=init, **kwargs)
  model.__name__ = 'VGG16'
  return model
  
def vgg19(input_size, init, **kwargs):
  """VGG 19-Layer model (configuration "E")
  
  Args:
    input_size: the input shape (width, height, channels)
  """
  in_linear = adaptive_linear(cfg['VGG']['E'], input_size)
  model = VGG(make_layers(cfg['VGG']['E'], input_size[2]), in_linear=in_linear, init_weights=init, **kwargs)
  model.__name__ = 'VGG19'
  return model

def lenet5(input_size, init, **kwargs):
  """Custom LeNet 5-Layer model (configuration "5")
  
  Args:
    input_size: the input shape (width, height, channels)
  """
  in_linear = adaptive_linear(cfg['LeNet']['5'], input_size)
  model = LeNet(make_layers(cfg['LeNet']['5'], input_size[2]), in_linear=in_linear, init_weights=init, **kwargs)
  model.__name__ = 'LeNet5'
  return model
  
def lenet8(input_size, init, **kwargs):
  """Custom LeNet 8-Layer model (configuration "8")
  
  Args:
    input_size: the input shape (width, height, channels)
  """
  in_linear = adaptive_linear(cfg['LeNet']['8'], input_size)
  model = LeNet(make_layers(cfg['LeNet']['8'], input_size[2]), in_linear=in_linear, init_weights=init, **kwargs)
  model.__name__ = 'LeNet8'
  return model
  
def lenet9(input_size, init, **kwargs):
  """Custom LeNet 9-Layer model (configuration "9")
  
  Args:
    input_size: the input shape (width, height, channels)
  """
  in_linear = adaptive_linear(cfg['LeNet']['9'], input_size)
  model = LeNet(make_layers(cfg['LeNet']['9'], input_size[2]), in_linear=in_linear, init_weights=init, **kwargs)
  model.__name__ = 'LeNet9'
  return model
  
def make_student(model_id, cfg, input_size, init, **kwargs):
  """Create student network given the teacher's
     configuration and number of input channels
  """
  in_linear = adaptive_linear(cfg, input_size)
  if model_id == 'LeNet5' or model_id == 'LeNet8':
    model = LeNetStudent(make_student_layers(cfg, input_size[2]), in_linear=in_linear, init_weights=init, **kwargs)
  else:
    model = VGGStudent(make_student_layers(cfg, input_size[2]), in_linear=in_linear, init_weights=init, **kwargs)
  model.__name__ = 'Student_' + model_id
  return model


def make_student_layers(cfg, in_channels=3):
  """Make student layers given configuration and
     number of input data channels
  """
  layers = []
  for v in cfg:
    if v == 'M':
      layers += [nn.MaxPool2d(kernel_size = 2, stride = 2)]
    else:
      conv2d = nn.Conv2d(in_channels, v * in_channels, kernel_size = 3, padding = 1, groups = in_channels, bias = True)
      bias = Bias(in_channels, v)
      layers += [conv2d, nn.ReLU(inplace = True), bias]
      in_channels = v
  return nn.Sequential(*layers)
  
def model_factory(model_id, dataset, init='x_gaussian', **kwargs):
  """Factory of student and teacher models. Useful to train
     student models from scratch
  """
  
  if dataset == 'mnist':
    input_size = (28, 28, 1)
  elif dataset == 'cifar10':
    input_size = (32, 32, 3)
  else:
    raise ValueError('Unsupported dataset: {}'.format(dataset))
  
  if model_id == 'VGG7':
    model = vgg7(input_size, init=init, **kwargs)
  elif model_id == 'VGG9':
    model = vgg9(input_size, init=init, **kwargs)
  elif model_id == 'VGG11exp':
    model = vgg11exp(input_size, init=init, **kwargs)
  elif model_id == 'VGG11':
    model = vgg11(input_size, init=init, **kwargs)
  elif model_id == 'VGG13':
    model = vgg13(input_size, init=init, **kwargs)
  elif model_id == 'VGG16':
    model = vgg16(input_size, init=init, **kwargs)
  elif model_id == 'VGG19':
    model = vgg19(input_size, init=init, **kwargs)
  elif model_id == 'LeNet5':
    model = lenet5(input_size, init=init, **kwargs)
  elif model_id == 'LeNet8':
    model = lenet8(input_size, init=init, **kwargs)
  elif model_id == 'LeNet9':
    model = lenet9(input_size, init=init, **kwargs)
  elif model_id == 'Student_VGG7':
    model = make_student('VGG7', cfg['VGG']['B1'], input_size=input_size, init=init, **kwargs)
  elif model_id == 'Student_VGG9':
    model = make_student('VGG9', cfg['VGG']['B2'], input_size=input_size, init=init, **kwargs)
  elif model_id == 'Student_VGG11exp':
    model = make_student('VGG11exp', cfg['VGG']['B3'], input_size=input_size, init=init, **kwargs)
  elif model_id == 'Student_VGG11':
    model = make_student('VGG11', cfg['VGG']['A'], input_size=input_size, init=init, **kwargs)
  elif model_id == 'Student_VGG13':
    model = make_student('VGG13', cfg['VGG']['B'], input_size=input_size, init=init, **kwargs)
  elif model_id == 'Student_VGG16':
    model = make_student('VGG16', cfg['VGG']['D'], input_size=input_size, init=init, **kwargs)
  elif model_id == 'Student_VGG19':
    model = make_student('VGG19', cfg['VGG']['E'], input_size=input_size, init=init, **kwargs)
  elif model_id == 'Student_LeNet5':
    model = make_student('LeNet5', cfg['LeNet']['5'], input_size=input_size, init=init, **kwargs)
  elif model_id == 'Student_LeNet8':
    model = make_student('LeNet8', cfg['LeNet']['8'], input_size=input_size, init=init, **kwargs)
  elif model_id == 'Student_LeNet9':
    model = make_student('LeNet9', cfg['LeNet']['9'], input_size=input_size, init=init, **kwargs)
  else:
    raise ValueError("No model found with key " + model_id) 
  return model
  
def teacher_factory(model_id, dataset, init='x_gaussian', **kwargs):
  """Dictionary of teacher models
  
  Args:
    model_id (str): the model identifier (e.g. 'vgg11')
    input_size (int tuple): the input dataset (width, height, channels)
  """
  if dataset == 'mnist':
    input_size = (28, 28, 1)
  elif dataset == 'cifar10':
    input_size = (32, 32, 3)
  else:
    raise ValueError('Unsupported dataset: {}'.format(dataset))
  
  if model_id == 'VGG7':
    model = vgg7(input_size, init=init, **kwargs)
  elif model_id == 'VGG9':
    model = vgg9(input_size, init=init, **kwargs)
  elif model_id == 'VGG11exp':
    model = vgg11exp(input_size, init=init, **kwargs)
  elif model_id == 'VGG11':
    model = vgg11(input_size, init=init, **kwargs)
  elif model_id == 'VGG13':
    model = vgg13(input_size, init=init, **kwargs)
  elif model_id == 'VGG16':
    model = vgg16(input_size, init=init, **kwargs)
  elif model_id == 'VGG19':
    model = vgg19(input_size, init=init, **kwargs)
  elif model_id == 'LeNet5':
    model = lenet5(input_size, init=init, **kwargs)
  elif model_id == 'LeNet8':
    model = lenet8(input_size, init=init, **kwargs)
  elif model_id == 'LeNet9':
    model = lenet9(input_size, init=init, **kwargs)
  else:
    raise ValueError("No model found with key " + model_id) 
  return model

def student_factory(model_id, dataset, init='x_gaussian', **kwargs):
  """Dictionary of student models
  
  Args:
    model_id (str): the model identifier (e.g. 'vgg11')
    input_size (int tuple): the input dataset (width, height, channels)
  """
  if dataset == 'mnist':
    input_size = (28, 28, 1)
  elif dataset == 'cifar10':
    input_size = (32, 32, 3)
  else:
    raise ValueError('Unsupported dataset: {}'.format(dataset))
    
  if model_id == 'VGG7':
    model = make_student(model_id, cfg['VGG']['B1'], input_size=input_size, init = init, **kwargs)
  elif model_id == 'VGG9':
    model = make_student(model_id, cfg['VGG']['B2'], input_size=input_size, init = init, **kwargs)
  elif model_id == 'VGG11exp':
    model = make_student(model_id, cfg['VGG']['B3'], input_size=input_size, init = init, **kwargs)
  elif model_id == 'VGG11':
    model = make_student(model_id, cfg['VGG']['A'], input_size=input_size, init = init, **kwargs)
  elif model_id == 'VGG13':
    model = make_student(model_id, cfg['VGG']['B'], input_size=input_size, init = init, **kwargs)
  elif model_id == 'VGG16':
    model = make_student(model_id, cfg['VGG']['D'], input_size=input_size, init = init, **kwargs)
  elif model_id == 'VGG19':
    model = make_student(model_id, cfg['VGG']['E'], input_size=input_size, init = init, **kwargs)
  elif model_id == 'LeNet5':
    model = make_student(model_id, cfg['LeNet']['5'], input_size=input_size, init = init, **kwargs)
  elif model_id == 'LeNet8':
    model = make_student(model_id, cfg['LeNet']['8'], input_size=input_size, init = init, **kwargs)
  elif model_id == 'LeNet9':
    model = make_student(model_id, cfg['LeNet']['9'], input_size=input_size, init = init, **kwargs)
  else:
    raise ValueError("No model found with key " + model_id) 
  return model
  
class Bias(nn.Module):
  """Sums along channel dimension and adds learnable bias
  """
  
  def __init__(self, in_channels, out_channels, bias = False):
    super(Bias, self).__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    if bias:
      self.bias = nn.Parameter(torch.zeros(1, self.out_channels, 1, 1))
    else:
      self.bias = None
    
  def forward(self, x):
    x = x.view(x.size(0), self.in_channels, -1, x.size(2), x.size(3))
    x = x.sum(1) # sum across input channels
    if self.bias is not None:
      x = x + self.bias
    return x

