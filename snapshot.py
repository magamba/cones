# -*- coding: utf-8 -*-

import os
import logging
import json

import torch
import torch.nn as nn
import dataparallel
import models

"""Utilities to save and load models
"""

def save_model(net, filename, dirname):
  """Save a model indexed to the specified path.
  """
  logger = logging.getLogger('train')
  
  path = os.path.join(os.path.normpath(dirname), filename)
  logger.info('Saving model {} to {}'.format(net.__name__, path))
  if isinstance(net, torch.nn.DataParallel):
    torch.save(net.module.state_dict(), path)
  else:
    torch.save(net.state_dict(), path)

def load_model(model_name, dataset, path, device):
  """Load a model from file for inference.
  
  Keyword arguments:
  model_name (str) -- name of the model architecture
  dataset (int) -- dataset (used to infer input dimensionality)
  path (str) -- path to the saved model
  device (torch.device) -- where to move the model after loading
  """
  
  logger = logging.getLogger('train')
  
  net = models.model_factory(model_name, dataset)

  # load parameters
  logger.info('Loading model {} from {}'.format(net.__name__, path))
  net.load_state_dict(torch.load(path), map_location=device)

  # move to device
  net = net.to(device = device)

  # set model to inference mode
  net = net.eval()

  return net

def load_student(args, device, legacy=True):
  """Load student network
  """
  
  logger = logging.getLogger('nesting')
  
  student = models.student_factory(args.arch, args.dataset)
  #student = nn.DataParallel(student)
  
  student = student.to(device=device)
  
  # load student from file
  if os.path.isfile(args.student):
    logger.info("Loading student network from {}".format(args.student))
    
    state_dict = torch.load(args.student, map_location=device)
    if legacy:
      state_dict = state_dict()
    
    if isinstance(student, torch.nn.DataParallel):
      student.module.load_state_dict(state_dict)
    else:
      student.load_state_dict(state_dict)
  else:
    raise ValueError('Missing student model definition. Specify it with --student [FILENAME]')
  
  student = student.eval()
  return student

def save_snapshot(net, optimizer, scheduler, epoch, dirname, init_scheme=None):
  """Save snapshot of training
  """
  logger = logging.getLogger('train')
  
  if init_scheme is not None:
    filename = net.__name__ + '_' + str(epoch) + '_' + str(init_scheme)  + '.tar'
  else:
    filename = net.__name__ + '_' + str(epoch) + '.tar'
  path = os.path.join(dirname, filename)
  
  model_state_dict = {}
  if isinstance(net, torch.nn.DataParallel):
    model_state_dict = net.module.state_dict()
  else:
    model_state_dict = net.state_dict()
   
  state_dictionary = {
            'epoch': epoch,
            'model_state_dict' : model_state_dict,
            'optimizer_state_dict' : optimizer.state_dict(),
            'scheduler_state_dict' : {}
            }
  if scheduler is not None:
    state_dictionary['scheduler_state_dict'] = scheduler.state_dict()
    
  logger.info('Saving snapshot of {}, epoch {} to {}'.format(net.__name__, epoch, path))
  torch.save(state_dictionary, path)


def load_snapshot(net, optimizer, scheduler, filename, device):
  """Load a stored snapshot
  
  net (nn.Module) -- an model instance
  optimizer -- optimizer instance
  scheduler -- scheduler instance
  filename (str) -- path to the stored model (.tar)
  device (torch.device) -- device where to load the model to
  
  """
  logger = logging.getLogger('train')
  logger.info('Loading snapshot of {} from {}'.format(net.__name__,filename))
  
  checkpoint = torch.load(filename, map_location=device)
  if isinstance(net, torch.nn.DataParallel):
    net.module.load_state_dict(checkpoint['model_state_dict'])
  else:
    net.load_state_dict(checkpoint['model_state_dict'])
  if optimizer is not None:
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    optimizer.zero_grad()
  if scheduler is not None:
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
  epoch = checkpoint['epoch']
  
  net = net.train()
  
  return net, optimizer, scheduler, epoch
  

def save_results(results, args):
  """Save a dictionary of results to json file
  """
  result_path = args.path
  run_name = os.path.split(os.path.split(args.student)[0])[1]
  dirname = os.path.join(result_path, args.dataset)
  filename = os.path.splitext(os.path.basename(args.student))[0]  
  result_path = os.path.join(dirname, run_name)
  
  if not os.path.exists(result_path):
    os.makedirs(result_path)
  filename = os.path.join(result_path, filename) + '.json'

  with open(filename, 'wb') as fp:
    fp.write(json.dumps(results).encode("utf-8"))

  
