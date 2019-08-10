# -*- coding: utf-8 -*-

import logging
import os

def print_results_check(results, logger_name=None):
  """Basic sanity check of compute results
  """
  
  msg = "\nResults dictionary:\n"
  
  # check number of conv blocks
  num_blocks = len(results["blocks"])
  msg = msg + "\tnumber of convolutional blocks: {}\n".format(num_blocks)
  
  for block_id, block in enumerate(results["blocks"]):
    msg = msg + "\tconvolutional block {}:\n".format(block_id+1)
    
    # check number of layer pairs
    msg = msg + "\t\tnumber of layer pairs: {}\n".format(len(results["blocks"][block_id]["layer_pairs"]))
    
    for pair_id, pair in enumerate(results["blocks"][block_id]["layer_pairs"]):
      # check number of output filters
      num_pairs = len(results["blocks"][block_id]["layer_pairs"][pair_id])
      msg = msg + "\t\t\t pair {}, number of filters: {}\n".format(pair_id+1, num_pairs)
      
  if logger_name is not None:
    logger = logging.getLogger(logger_name)
    logger.info(msg)
  else:
    print(msg)  

def print_json(json_dict):
  """Pretty prints a json dictionary
  """
  import json
  
  parsed = json.loads(json.dumps(json_dict).encode("utf-8"))
  print(json.dumps(parsed, indent=2, sort_keys=True))

def prepare_dirs(args):
  """Parse user options and prepare
     directories and logger
  """
  path = args.path
  logdir = './log'
  logdir = os.path.join(
              os.path.join(logdir,'nesting'),
              args.dataset)
  if args.noisy > 0:
    noisestr = 'noise_' + str(int(args.noisy * 100))
    logdir = os.path.join(logdir, noisestr)
    
  logdir = os.path.join(logdir,args.arch)
  if not os.path.exists(path):
    os.mkdir(path)
  if not os.path.exists(logdir):
    os.makedirs(logdir)
  logfile = os.path.join(logdir, args.log)
  
  # init logging
  logger = logging.getLogger('nesting')
  logger.setLevel(logging.DEBUG)
  f_handler = logging.FileHandler(logfile)
  f_handler.setLevel(logging.DEBUG)
  c_handler = logging.StreamHandler()
  c_handler.setLevel(logging.ERROR)
  
  formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
  f_handler.setFormatter(formatter)
  c_handler.setFormatter(formatter)
  logger.addHandler(f_handler)
  logger.addHandler(c_handler)
    
  logger.info('Saving results to {}'.format(path))

def print_model_config(user_options, start_epoch):
  """ Print the model configuration defined by the user.
  
  Args
    user_options -- arg parser containing user options
  """
  logger = logging.getLogger('train')
  
  model_config_str = "\nModel configuration: \n"+ \
                   "\t architecture: {}\n".format(user_options.arch) + \
                   "\t dataset: {}".format(user_options.dataset) + \
                   " with noise probability {}\n".format(user_options.noisy) if user_options.noisy > 0 else "\n"
  model_config_str = model_config_str + \
                   "\t initialization: {}\n".format(user_options.init) + \
                   "\t start epoch: {}\n".format(start_epoch) + \
                   "\t epochs: {}\n".format(user_options.epochs) + \
                   "\t batch size: {}\n".format(user_options.batch_size) + \
                   "\t base lr: {:.5f}\n".format(user_options.lr) + \
                   "\t loss: {}\n".format(user_options.loss) + \
                   "\t optimizer: {}\n".format(user_options.optimizer)
  if user_options.optimizer == 'sgd':
    model_config_str = model_config_str + \
                   "\t momentum: {}\n".format(user_options.momentum) + \
                   "\t weight decay: {}\n".format(user_options.weight_decay) + \
                   "\t lr rescaled by {} after {} steps\n".format(user_options.lr_decay, user_options.lr_step)
  model_config_str = model_config_str +  \
                     "\t saving snapshots every {} epochs to {}\n".format(user_options.snapshot_every, user_options.path)
  logger.info(model_config_str)
  
def print_student_config(user_options):
  """Print the training configuration for knowledge distillation
  """
  logger = logging.getLogger('train')
  
  model_config_str = "\nModel configuration: \n"+ \
                   "\t teacher architecture: {}\n".format(user_options.arch) + \
                   "\t teacher model loaded from {}\n".format(user_options.resume_from) + \
                   "\t dataset: {}".format(user_options.dataset) + \
                   " with noise probability {}\n".format(user_options.noisy)
  model_config_str = model_config_str + \
                   "\t initialization: {}\n".format(user_options.init) + \
                   "\t epochs: {}\n".format(user_options.epochs) + \
                   "\t base lr: {:.3f}\n".format(user_options.lr) + \
                   "\t optimizer: {}\n".format(user_options.optimizer) + \
                   "\t softmax temperature: {}\n".format(user_options.temperature) + \
                   "\t alpha: {}\n".format(user_options.alpha)
  if user_options.optimizer == 'sgd':
    model_config_str = model_config_str + \
                   "\t momentum: {}\n".format(user_options.momentum) + \
                   "\t weight decay: {}\n".format(user_options.weight_decay) + \
                   "\t lr rescaled by {} after {} steps\n".format(user_options.lr_decay, user_options.lr_step)
  model_config_str = model_config_str +  \
                     "\t saving snapshots every {} epochs to {}\n".format(user_options.snapshot_every, user_options.path)
  logger.info(model_config_str)
  
def print_val_loss(epoch, avg_loss, accuracy):
  """ Print the average validation loss and accuracy
  """
  logger = logging.getLogger('train')
  logger.info('\t epoch: {}, test loss: {:.6f}, acc: {:.3f}'.format(
        epoch, avg_loss, accuracy))
        
def print_train_loss(epoch, avg_loss, batch_idx, num_batches):
  """Print the running average of the train loss for the given batch
  """
  logger = logging.getLogger('train')
  logger.info('\t epoch: {}, batch: {}/{}, train loss: {:.6f}'.format(
          epoch, batch_idx+1, num_batches, avg_loss))
          
def print_train_loss_epoch(epoch, epoch_loss):
  """Print the average train loss for the specified epoch
  """
  logger = logging.getLogger('train')
  logger.info('\t epoch: {}, train loss: {:.6f}'.format(
          epoch, epoch_loss))
