# -*- coding: utf-8 -*-

""" Compute nesting along paths for a given network
"""

import os
import sys
import logging
import torch
import torch.nn as nn

from constants import OUT_PARTIAL_IN, OUT_FULL_IN, IN_PARTIAL_OUT, IN_FULL_OUT
import models
import snapshot
import linalg
import utils

from mpi4py import MPI
import numpy as np

def binary_nesting(filter_dict):
  """Compute binary nesting between 
     a convolutional channel and the corresponing
     input filters
  """  
  # compute covering cone for input filters
  apex_distance = np.max(filter_dict["distance"])
  opening_distance = np.min(filter_dict["opening"])
  
  if apex_distance >= 0: # cone_2 inside cone_1
    if opening_distance > 0: # partial nesting
      return OUT_PARTIAL_IN
    else:
      return OUT_FULL_IN # full nesting
  else: # cone_1 inside cone_2
    if opening_distance >= 0:
      return IN_FULL_OUT # full nesting
    else:
      return IN_PARTIAL_OUT # part nesting

def new_dictionary(args):
  """Create empty dictionary for storing
     the results
  """
  name = os.path.splitext(os.path.basename(args.student))[0]
  run = (os.path.split(os.path.split(args.student)[0])[1]).split('_')[1]
  epoch = name.split('_')[2] # student network only
  
  results_dictionary = {
    "name" : name,
    "arch" : args.arch,
    "dataset" : args.dataset,
    "epoch" : epoch,
    "run" : run,
    "blocks" : [] # list of convolutional blocks
  }
  return results_dictionary

def new_block_dict():
  """Dictionary of results per convolutional block
  """
  block_dict = { 
        "layer_pairs" : [] # each item is a list of filters
      }
  return block_dict
  
def new_filter():
  """Dictionary of results for a kernel
  """
  kernel = {
    "distance" : [],
    "opening" : [],
    "rotation" : [],
    "binary_nesting" : ""
  }
  return kernel

def next_block(features, last = 0):
  """Return the indices of pairs of layers at the next block
     current (tuple) - indices l and l+1 of the current layer
  """
  indices = [] # pairs of consecutive layers within a block
  if last >= len(features)-1:
    return []
  for layer_id, feature in enumerate(features):
    if layer_id < last:
      continue
    if isinstance(feature, nn.Conv2d):
      indices.append(layer_id)
    if isinstance(feature, nn.MaxPool2d):
      break
  if len(indices) < 2: # no pairs found within this block
    return next_block(features, layer_id +1)
  else:
    pairs = [(indices[i], indices[i+1]) for i in range(len(indices)-1)]
  return pairs
  
def update_input_size(input_size, logger_name=None):
  """ Update input spatial size after max pooling
      assuming 3x3 convolutions, 1 zero padding, stride 1 
      and pooling of (2,2) with stride 2
  """
#  if input_size[2] == 6:
#    channel = 16
#  else:
#    channel = min(input_size[2] * 2, 512)  
  if logger_name is not None:
    logger = logging.getLogger(logger_name)
    
    msg = "Updating input_size, before: {}".format(input_size)
    input_size = (input_size[0] // 2, input_size[1] // 2, input_size[2])
    msg = msg + " after: {}".format(input_size)
    
    logger.debug(msg)
  else:
    input_size = (input_size[0] // 2, input_size[1] // 2, input_size[2])
  
  return input_size
  
def compute_nesting(args, comm, device):
  """Split data between processes
     gathers results and saves them to file
  """
  rank = comm.Get_rank()
  size = comm.Get_size()
  
  logger = logging.getLogger('nesting')
  
  if args.dataset == 'mnist':
    input_size = (28,28,1)
  elif args.dataset == 'cifar10':
    input_size = (32,32,3)
  else:
    raise ValueError('Unknown dataset {}'.format(args.dataset))
  
  logger.info("Precomputing rotations...")
  rotations_dict = linalg.init_rotations([32, 16, 8, 4, 2])
  
  if rank == 0:
    
    logger.info('Running with {} processes.'.format(size))
    
    logger.info('Loading {} on {}.'.format(args.student, args.dataset))
    if (os.path.splitext(os.path.basename(args.student))[1]) == '.pt':
      student = snapshot.load_student(args, device, legacy=args.legacy)
    else:
      student = models.student_factory(args.arch, args.dataset)
      student = student.to(device)
      student, _, _, _ = snapshot.load_snapshot(student, None, None, args.student, device)
      
    result_dictionary = new_dictionary(args)
    
    if isinstance(student, nn.DataParallel):
      params = list(student.module.features.parameters())
      features = student.module.features
    else:
      params = list(student.features.parameters())
      features = student.features
      
    # iterate over blocks
    layer_pairs = next_block(features, 0)
    layer_offset = 2 # offset between the conv params of two layers in param
    block_id = 0
    block_offset = 0 # start index for the params of the next conv block
    
    while len(layer_pairs) > 0:
          
      logger.info('Iterating through convolutional block {}.'.format(block_id))
    
      # create new block dictionary
      block_dict = new_block_dict()
      
      logger.info('Found {} pairs of consecutive layers'.format(len(layer_pairs)))
      
      for index, pair in enumerate(layer_pairs):
        
        logger.info('Extracting features for layer pair {}'.format(index))
        
        # load list of filters from layer l+1
        param_id = index * layer_offset + block_offset
        out_filters = params[param_id+2]
        out_biases = params[param_id+3]
        # convert filters to numpy
        out_filters = np.squeeze(out_filters.detach().cpu().numpy()) # [4096,3,3]
        out_biases = np.squeeze(out_biases.detach().cpu().numpy()) # [4096,]
        
        if out_filters.shape[0] % size != 0:
          logger.warning("Warning, the number of processes does not divide the number of output filters, the remainder will be left out.")
        
        data = np.array([out_filters.shape[0] // size, out_filters.shape[1], out_filters.shape[2]], dtype='i')
        
        comm.Barrier()
        comm.Bcast(data, root=0) # broadcast number of out_filters per process
        
        logger.debug('Filling buffers...')
        
        shape = data[0] * data[1] * data[2]
        sendbuf_ofilters = np.empty([size, shape], dtype='f')
        sendbuf_obiases = np.empty([size, out_biases.shape[0] // size], dtype='f')
        
        sendbuf_ofilters = np.reshape(out_filters, (size, -1))
        sendbuf_obiases = np.reshape(out_biases, (size, -1))
        
        recvbuf_ofilters = np.empty(shape, dtype='f')
        recvbuf_obiases = np.empty(out_biases.shape[0] // size, dtype='f')
        
        comm.Scatter(sendbuf_ofilters, recvbuf_ofilters, root=0)
        comm.Scatter(sendbuf_obiases, recvbuf_obiases, root=0)
        
        # for each output filter
        # reshape weight and bias
        ofilters = np.reshape(recvbuf_ofilters, (data[0], data[1], data[2]))
        obiases = np.reshape(recvbuf_obiases, (data[0], -1))
        
        # load list of filters from layer l
        # convert to numpy
        in_filters = np.squeeze(params[param_id].detach().cpu().numpy())
        in_biases = np.squeeze(params[param_id+1].detach().cpu().numpy())
        
        # populate buffers
        buf_index = input_size[2] # n0
        in_kernels = in_filters.shape[0] // buf_index # n1
        step = in_kernels // size # n1 / size
        
        sendbuf_ifilters = np.array([[in_filters[in_kernels *i + rank * step +j, :] for i in range(buf_index)] for j in range(step)], dtype='f')
        sendbuf_ibiases = np.array([[in_biases[in_kernels *i + rank * step +j] for i in range(buf_index)] for j in range(step)], dtype='f')
        
        for r in range(1, size):
          sendbuf_ifilters = np.concatenate((sendbuf_ifilters, np.array([[in_filters[in_kernels * i + r * step +j, :] for i in range(buf_index)] for j in range(step)], dtype='f')),0)
          sendbuf_ibiases = np.concatenate((sendbuf_ibiases, np.array([[in_biases[in_kernels * i + r * step +j] for i in range(buf_index)] for j in range(step)], dtype='f')),0)
              
        # reshape input filter matrix
        np.reshape(sendbuf_ifilters,(size, -1))
        # reshape input biases
        np.reshape(sendbuf_ibiases,(size, -1))
        
        # broadcast input filter shape and number of channels per process
        idata = np.array([in_filters.shape[0] // size, in_filters.shape[1], in_filters.shape[2], step], dtype='i')
        comm.Bcast(idata, root=0)
        comm.Barrier()
        
        shape = idata[0] * idata[1] * idata[2]
        recvbuf_ifilters = np.empty(shape, dtype='f')
        recvbuf_ibiases = np.empty(idata[0], dtype='f')
        
        # Scatter
        comm.Scatter(sendbuf_ifilters, recvbuf_ifilters, root=0)
        comm.Scatter(sendbuf_ibiases, recvbuf_ibiases, root=0)
        
        # for each input filter
        # reshape weight and bias        
        ifilters = np.reshape(recvbuf_ifilters, (idata[0], idata[1], idata[2]))
        ibiases = np.reshape(recvbuf_ibiases, (idata[0], -1))
        
        # update input channels
        input_size = (input_size[0], input_size[1], in_kernels)
                
        logger.info('Computing pairwise nesting')
        
        # buffer for receiving results
        result_buffer = None
        out_result = []
        
        channels_per_proc = int(idata[3])
        out_kernels = int(ofilters.shape[0] // channels_per_proc)
        in_channels = int(ifilters.shape[0] // channels_per_proc)
        for out in range(ofilters.shape[0]):
          filter_dict = new_filter()
          # take output filter and bias
          b1 = obiases[out]
          w1 = ofilters[out,:]
          
          in_step = out // out_kernels # base index of the corresponding in_channels        
          # for each input filter
          for inp in range(in_step * in_channels, (in_step +1) * in_channels):
            # weight and bias
            b2 = ibiases[inp]
            w2 = ifilters[inp, :]
            
            dist, shear, rot = linalg.pairwise_nesting(w1, b1, w2, b2, input_size, rotations_dict, args.quick)
            filter_dict["distance"].append(dist)
            filter_dict["opening"].append(shear)
            filter_dict["rotation"].append(rot)
          
          filter_dict["binary_nesting"] = binary_nesting(filter_dict)
          filter_dict["distance"] = np.mean(np.abs(filter_dict["distance"])).item()
          filter_dict["opening"] = np.mean(np.abs(filter_dict["opening"])).item()
          filter_dict["rotation"] = np.mean(np.abs(filter_dict["rotation"])).item()
          # store
          out_result.append(filter_dict)
          
        # receive results
        logger.info('Collecting results...')
        result_buffer = comm.gather(out_result, root=0)
        
        # merge lists of filters
        for i in range(1, len(result_buffer)):
          out_result += result_buffer[i]
          
        # add to dict
        block_dict["layer_pairs"].append(out_result)
        if index == len(layer_pairs) -1:
          result_dictionary["blocks"].append(block_dict)
      
      # update block indices    
      block_id = block_id + 1
      block_offset = block_offset + layer_offset * (len(layer_pairs) +1)
      
      # change input channels to n2
      input_size = (input_size[0], input_size[1], out_kernels)
      
      # read next block
      last = layer_pairs[len(layer_pairs)-1][1] +1
      layer_pairs = next_block(features, last)
      
      # assuming 3x3 convolutions, 1 zero padding, stride 1 
      # and pooling of (2,2) with stride 2
      input_size = update_input_size(input_size)
      
      if (len(layer_pairs)) == 0:
        input_size = (0, 0, 0) # send exit command to all processes
      
      input_size = comm.bcast(input_size, root = 0)
      comm.Barrier()
      if input_size[0] < 2 or input_size[1] < 2:
        break
  
  else:
  
    while input_size[0] > 1 and input_size[1] > 1:
      data = np.empty((3), dtype='i')
      
      comm.Barrier()
      comm.Bcast(data, root=0) # broadcast number of out_filters per process
      
      shape = data[0] * data[1] * data[2]
      
      sendbuf_obiases = None
      sendbuf_ofilters = None
      
      recvbuf_ofilters = np.empty(shape, dtype='f')
      recvbuf_obiases = np.empty(data[0], dtype='f')
      
      comm.Scatter(sendbuf_ofilters, recvbuf_ofilters, root=0)
      comm.Scatter(sendbuf_obiases, recvbuf_obiases, root=0)
      
      # for each output filter
      # reshape weight and bias
      ofilters = np.reshape(recvbuf_ofilters, (data[0], data[1], data[2]))
      obiases = np.reshape(recvbuf_obiases, (data[0], -1))
      
      sendbuf_ifilters = None
      sendbuf_ibiases = None
      
      # input filters shape
      idata = np.empty((4), dtype='i')
      comm.Bcast(idata, root=0)
      comm.Barrier()
      
      shape = idata[0] * idata[1] * idata[2]
      recvbuf_ifilters = np.empty(shape, dtype='f')
      recvbuf_ibiases = np.empty(idata[0], dtype='f')
      
      # Scatter
      comm.Scatter(sendbuf_ifilters, recvbuf_ifilters, root=0)
      comm.Scatter(sendbuf_ibiases, recvbuf_ibiases, root=0)
     
      # for each input filter
      # reshape weight and bias      
      ifilters = np.reshape(recvbuf_ifilters, (idata[0], idata[1], idata[2]))
      ibiases = np.reshape(recvbuf_ibiases, (idata[0], -1))
      
      # buffer for receiving results
      out_result = []
      
      channels_per_proc = int(idata[3])
      out_kernels = int(ofilters.shape[0] // channels_per_proc)
      in_channels = int(ifilters.shape[0] // channels_per_proc)
      for out in range(ofilters.shape[0]):
        # take output filter and bias
        b1 = obiases[out]
        w1 = ofilters[out,:]
        
        filter_dict = new_filter()
        
        in_step = out // out_kernels # base index of the corresponding in_channels        
        # for each input filter
        for inp in range(in_step * in_channels, (in_step +1) * in_channels):
          # weight and bias
          b2 = ibiases[inp]
          w2 = ifilters[inp, :]
          
          dist, shear, rot = linalg.pairwise_nesting(w1, b1, w2, b2, input_size, rotations_dict, args.quick)
          filter_dict["distance"].append(dist)
          filter_dict["opening"].append(shear)
          filter_dict["rotation"].append(rot)
        
        filter_dict["binary_nesting"] = binary_nesting(filter_dict)
        filter_dict["distance"] = np.mean(np.abs(filter_dict["distance"])).item()
        filter_dict["opening"] = np.mean(np.abs(filter_dict["opening"])).item()
        filter_dict["rotation"] = np.mean(np.abs(filter_dict["rotation"])).item()
        # store
        out_result.append(filter_dict)
        
      # receive results
      out_result = comm.gather(out_result, root=0)
      
       # wait for input_size
      input_size = comm.bcast(input_size, root=0)
      comm.Barrier()
  
  if rank == 0:
    logger.info('Finishing...')
  else:
    result_dictionary = None
  return result_dictionary

def main(args, comm):

  global device
 
  if args.cuda and torch.cuda.is_available():
    device = torch.device('cuda:0')
  else:
    device = torch.device('cpu')
  
  # compute nesting
  result_dictionary = compute_nesting(args, comm, device)
  
  if comm.Get_rank() == 0:
    utils.print_results_check(result_dictionary, logger_name='nesting')
    snapshot.save_results(result_dictionary, args)

def parse_arguments(comm):
  """Parse command line options and broadcast to
     all processes. Exit gracefully on malformed
     options.
  """
  
  default_path = "./results"

  # parse command line arguments
  import argparse
  parser = argparse.ArgumentParser()

  # datasets
  parser.add_argument("--arch", type=str, default='VGG13', help="The teacher network architecture.")
  parser.add_argument("--dataset", type=str, default='cifar10', help="Available datasets: mnist, cifar10.")
  parser.add_argument("--dataset-path", type=str, default='./data', help="Path where datasets are stored.")
  # noisy labels
  parser.add_argument("--noisy", type=float, default=0, help="Percentage of corrupted labels [default = 0, max = 100].")
  
  # path where to store results
  parser.add_argument("--path", type=str, default=default_path, help="The dirname where to store results [default = './results'].")
  
  # path to a student network
  parser.add_argument("--student", type=str, default='', help="Path to a student snapshot [default = None].")
  
  # log file
  parser.add_argument("--log", type=str, default='nesting.log', help="Logfile name [default = 'nesting.log'].")

  # use cuda
  parser.add_argument("--cuda", default=False, action='store_true', help="Wheter to load the model on GPU.")
  
  # load legacy models (see snapshot.load_student())
  parser.add_argument("--legacy", default=False, action='store_true', help="Load legacy snapshots, stored with early versions of train.py.")
  
  # skip rotations
  parser.add_argument("--quick", default=False, action='store_true', help="Skip rotations, for faster runs.")

  args = None
  
  try:
    if comm.Get_rank() == 0:
      # parse the command line
      args = parser.parse_args()
  finally:
    args = comm.bcast(args, root=0)
    
  if args is None:
    comm.Abort()
  return args

if __name__ == '__main__':
  import signal
  signal.signal(signal.SIGINT, lambda x, y: sys.exit(0))
  
  # init MPI
  comm = MPI.COMM_WORLD
  
  args = parse_arguments(comm)
  
  if comm.Get_rank() == 0:
    utils.prepare_dirs(args)
  
  main(args, comm)
