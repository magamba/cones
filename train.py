# -*- coding: utf-8 -*-

import sys
import os

import logging

import torch
import torch.nn as nn
import torch.optim as optim

import models
from dataparallel import NamedDataParallel
import data_loaders
import utils
import scores
import snapshot

import distillation

def load_optimizer(user_options, net):
  """Load the optimizer specified by user_options.
  """ 
  scheduler = None
  if user_options.optimizer == 'sgd':
    momentum = user_options.momentum
    step_size = user_options.lr_step
    optimizer = optim.SGD(net.parameters(), lr=user_options.lr, momentum=momentum, weight_decay = user_options.weight_decay)
    if step_size > 0:
      scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=user_options.lr_decay)
  elif user_options.optimizer == 'adam':
    optimizer = optim.Adam(net.parameters(), lr=user_options.lr)
  else:
    raise ValueError('Optimizer not supported.')
  return optimizer,  scheduler

def load_criterion(user_options):
  """Load the score function specified by user_options.
  """
  if user_options.loss == "softmax":
    criterion = nn.CrossEntropyLoss()
  elif user_options.loss == 'KD':
    criterion = distillation.KDLoss(user_options.temperature, user_options.alpha)
  else:
    raise ValueError('Unsupported score function.')
  return criterion

def train(model, end_epoch, train_loader, optimizer, criterion, scheduler, device, start_epoch = 0, snapshot_every = 0, test_loader = None, kill_plateaus = False, init_scheme=None):
  """Train the specified model according to user options.
  
    Args:
    
    model (nn.Module object) -- the model to be trained
    end_epoch (int) -- maximum number of epochs
    train_loader (object, DataLoader) -- train set loader
    optimizer (torch.optim optimizer) -- the optimizer to use
    criterion -- loss function to use
    scheduler -- learning rate scheduler
    device (torch.device) -- device to use
    start_epoch (int) -- starting epoch (useful for resuming training)
    snapshot_every (int) -- frequency of snapshots (in epochs)
    test_loader (optional, DataLoader) -- test set loader
    
  """
  if snapshot_every < 1:
    snapshot_every = end_epoch
  start_loss = 0.
  converged = True
  for epoch in range(start_epoch, end_epoch):
    # training loss
    avg_loss = 0
    epoch_loss = 0
    for batch_idx, (x, target) in enumerate(train_loader):
      optimizer.zero_grad()
      x, target = x.to(device=device), target.to(device=device)
      out = model(x)
      loss = criterion(out, target)
      avg_loss = avg_loss * 0.99 + loss.item() * 0.01
      epoch_loss += loss.item()
      loss.backward()
      optimizer.step()
      with torch.no_grad():
        if kill_plateaus:
          if epoch == start_epoch and batch_idx == 99:
            start_loss = avg_loss
          if epoch == 19 and batch_idx == 99:
            if scores.loss_plateaus(start_loss, avg_loss):
              logger.debug("Start loss: {}, current loss: {}. Model unlikely to converge. Quitting.".format(start_loss, avg_loss))
              converged = False
              return model, converged
      # report training loss
      if ((batch_idx+1) % 100 == 0) or ((batch_idx+1) == len(train_loader)):
        utils.print_train_loss(epoch, avg_loss, batch_idx, len(train_loader))
    # report training loss over epoch
    epoch_loss /= len(train_loader)
    utils.print_train_loss_epoch(epoch, epoch_loss)
    if scheduler is not None:
      scheduler.step()
    if ((epoch +1) % snapshot_every == 0) or ((epoch +1) == end_epoch):
      if test_loader is not None:
        val_loss, accuracy = scores.test(model, test_loader, criterion, device)
        utils.print_val_loss(epoch, val_loss, accuracy)
        model = model.train()
      # save snapshot
      snapshot.save_snapshot(model, optimizer, scheduler, epoch, snapshot_dirname, init_scheme)
  return model, converged
  
def distill(student, teacher, end_epoch, train_loader, optimizer, criterion, scheduler, tdevice, device, start_epoch = 0, snapshot_every = 0, kill_plateaus = False):
  """Train the specified model according to user options.
  
    Args:
    
    student (nn.Module) -- the student model to be trained
    teacher (nn.Module) -- the teacher model
    end_epoch (int) -- maximum number of epochs
    train_loader (object, DataLoader) -- train set loader
    optimizer (torch.optim optimizer) -- the optimizer to use
    criterion -- loss function to use (KD)
    scheduler -- learning rate scheduler
    tdevice -- device to use of the teacher network
    device (torch.device) -- device to use for the student network
    start_epoch (int) -- starting epoch (useful for resuming training)
    snapshot_every (int) -- frequency of snapshots (in epochs)
    test_loader (optional, DataLoader) -- test set loader
    
  """  
  if snapshot_every < 1:
    snapshot_every = end_epoch
  start_loss = 0.
  converged = True
  for epoch in range(start_epoch, end_epoch):
    # training loss
    avg_loss = 0
    epoch_loss = 0
    for batch_idx, (x, target) in enumerate(train_loader):
      optimizer.zero_grad()
      xcopy = x.clone().to(device=tdevice)
      x, target = x.to(device=device), target.to(device=device)
      with torch.no_grad():
        teacher_logits = teacher(xcopy)
      teacher_logits = teacher_logits.detach().to(device=device)
      out = student(x)
      loss = criterion(out, target, teacher_logits)
      avg_loss = avg_loss * 0.99 + loss.item() * 0.01
      epoch_loss += loss.item()
      loss.backward()
      optimizer.step()
      
      with torch.no_grad():
        if kill_plateaus:
          if epoch == start_epoch and batch_idx == 99:
            start_loss = avg_loss
                        
          if epoch == 19 and batch_idx == 99:
            if scores.loss_plateaus(start_loss, avg_loss):
              logger.debug("Start loss: {}, current loss: {}. Model unlikely to converge. Quitting.".format(start_loss, avg_loss))
              converged = False
              return student, converged
      # report training loss
      if ((batch_idx+1) % 100 == 0) or ((batch_idx+1) == len(train_loader)):
        utils.print_train_loss(epoch, avg_loss, batch_idx, len(train_loader))
    # report training loss over epoch
    epoch_loss /= len(train_loader)
    utils.print_train_loss_epoch(epoch, epoch_loss)
    if scheduler is not None:
      scheduler.step()
  return student, converged
  
def student_train_test(user_options):
  """Train student network by knowledge distillation
  
  Args
  run (int): the current independent run (used filenames)
  user_options (argparser) : user specified options
  """
  # get logger
  logging.getLogger('train')

  # load teacher model
  teacher = models.model_factory(user_options.arch, dataset=user_options.dataset, init=user_options.init)
  
  if torch.cuda.device_count() > 1:
    logger.info("Running teacher network on {} GPUs".format(torch.cuda.device_count()))
    teacher = NamedDataParallel(teacher)
    tdevice = device
  else:
    tdevice = torch.device('cpu')
  
  # move net to device
  teacher = teacher.to(device=tdevice)
  
  # load teacher network from file
  if os.path.isfile(user_options.resume_from):
    teacher, _, _, _ = snapshot.load_snapshot(teacher, None, None, user_options.resume_from, tdevice)
    teacher = teacher.eval()
  else:
    raise ValueError('Missing teacher model definition. Specify it with --resume-from [FILENAME]')
  
  # get data loader for the specified dataset
  train_loader, test_loader = data_loaders.load_dataset(user_options.dataset, user_options.dataset_path, user_options.noisy, user_options.batch_size)
  
  # load student
  student = models.student_factory(user_options.arch, user_options.dataset, init=user_options.init)
  
  if torch.cuda.device_count() > 1:
    logger.info("Running student network on {} GPUs".format(torch.cuda.device_count()))
    student = NamedDataParallel(student)
    
  student = student.to(device=device)

  # load optimizer, scheduler
  optimizer, scheduler = load_optimizer(user_options, student)

  # define loss
  criterion = load_criterion(user_options)

  # print model configuration
  start_epoch = 0
  utils.print_student_config(user_options)
  
  # save model at initialization
  teacher_name = os.path.basename(user_options.resume_from)
  teacher_name = os.path.splitext(teacher_name)[0] # remove file extension
  teacher_name = teacher_name.split('_')[0]
  filename = 'Student_' + teacher_name + '_' + str(start_epoch) + '.pt'
  snapshot.save_model(student, filename, snapshot_dirname)

  # train the model
  student, converged = distill(student, teacher, user_options.epochs, train_loader, optimizer, criterion, scheduler, tdevice, device, start_epoch, snapshot_every = user_options.epochs, kill_plateaus = user_options.kill_plateaus)
  
  if test_loader is not None:
    test_criterion = nn.CrossEntropyLoss()
    val_loss, accuracy = scores.test(student, test_loader, test_criterion, device)
    utils.print_val_loss(user_options.epochs, val_loss, accuracy)

  # save final model
  if converged:
    teacher_name = os.path.basename(user_options.resume_from)
    teacher_name = os.path.splitext(teacher_name)[0] # remove file extension
    filename = 'Student_' + teacher_name + '.pt'
    snapshot.save_model(student, filename, snapshot_dirname)
    
  
def train_test_net(run, user_options):
  """Train and save a network accoring to user options
  
  Args
  run (int): the current independent run (used in filenames)
  user_options (argparser) : user specified options
  """

  # get logger
  logging.getLogger('train')

  #initialize model
  net = models.model_factory(user_options.arch, dataset=user_options.dataset, init=user_options.init)
  
  if torch.cuda.device_count() > 1:
    logger.info("Running on {} GPUs".format(torch.cuda.device_count()))
    net = NamedDataParallel(net)
  
  # move net to device
  net = net.to(device=device)
  
  # get data loader for the specified dataset
  train_loader, test_loader = data_loaders.load_dataset(user_options.dataset, user_options.dataset_path, user_options.noisy, user_options.batch_size)

  # define loss
  criterion = load_criterion(user_options)
  criterion = criterion.to(device)
  
  # resume training from snapshot if specified
  start_epoch = 0
  if os.path.isfile(user_options.resume_from):
    # resume training given state dictionary
    optimizer, scheduler = load_optimizer(user_options, net)
    net, optimizer, scheduler, start_epoch = snapshot.load_snapshot(net, optimizer, scheduler, user_options.resume_from, device)
    start_epoch = start_epoch + 1
  else:
    # define optimizer
    optimizer, scheduler = load_optimizer(user_options, net)

  # print model configuration
  logger.info("Running trial {} of {}".format(run+1, user_options.runs))
  utils.print_model_config(user_options, start_epoch)
  
  if start_epoch == 0: 
    filename = net.__name__ + '_' + str(start_epoch) + '_' + str(user_options.init) + '.pt'
    logger.info("Saving model initialization to {}".format(filename))
    snapshot.save_model(net, filename, snapshot_dirname)

  # train the model
  net, converged = train(net, user_options.epochs, train_loader, optimizer, criterion, scheduler, device, start_epoch, snapshot_every = user_options.snapshot_every, test_loader = test_loader, kill_plateaus = user_options.kill_plateaus, init_scheme=user_options.init)
  
  if test_loader is not None:
    val_loss, accuracy = scores.test(net, test_loader, criterion, device)
    utils.print_val_loss(user_options.epochs, val_loss, accuracy)
    net = net.train()

  # save final model
  if converged:
    filename = net.__name__ + '_' + str(user_options.epochs) + '_' + user_options.init + '.pt'
    snapshot.save_model(net, filename, snapshot_dirname)


def main(user_options):
  """Read the training arguments from the command line and train a model.

  Keyword arguments:
  user_options -- command line arguments specified by user
  """

  global snapshot_dirname, device

  logger = logging.getLogger('train')
  
  if torch.cuda.is_available():
    device = torch.device('cuda:0')
  else:
    device = torch.device('cpu')
    
  if not os.path.exists(user_options.dataset_path):
    os.mkdir(user_options.dataset_path)
  logger.info('Using {} for loading/storing datasets.'.format(user_options.dataset_path))
  
  student = False
  if user_options.loss == 'KD': #change conditions here if needed
    student = True

  if student:
    # student will be saved to the same folder as teacher network
    snapshot_dirname = os.path.dirname(user_options.resume_from)
  else:
    snapshot_dirname = os.path.join(user_options.path, user_options.dataset)
    if user_options.noisy > 0:
      noisestr = 'noise_' + str(int(user_options.noisy * 100))
      snapshot_dirname = os.path.join(snapshot_dirname, noisestr)
      
  if user_options.snapshot_every > 0:
    logger.info('Saving snapshots to {}'.format(snapshot_dirname))
  
  if not os.path.exists(snapshot_dirname):
    os.makedirs(snapshot_dirname)

  # independent runs  
  num_runs = user_options.runs
  if (student and user_options.runs > 1):
    logger.warning('Training student network, only single runs are supported, but {} were specified. Ignoring.'.format(user_options.runs))
    num_runs = 1
    
  if student: # knowledge distillation
    student_train_test(user_options)
  else: # standard training
    snapshot_root = snapshot_dirname
    start_run = user_options.start_run
    for run in range(start_run, num_runs):
      snapshot_dirname = os.path.join(snapshot_root, 'run_' + str(run))
      if not os.path.exists(snapshot_dirname):
        os.mkdir(snapshot_dirname)
      train_test_net(run, user_options)

if __name__ == "__main__":

  import signal
  signal.signal(signal.SIGINT, lambda x, y: sys.exit(0))

  default_path = "./models"

  # parse command line arguments
  import argparse
  parser = argparse.ArgumentParser()

  # models
  parser.add_argument("--arch", type=str, default=None, help="Network architecture to be trained. Run without this option to see a list of all supported archs.")
  parser.add_argument("--init", type=str, default='x_gaussian', help="Weight initialization scheme [default = x_gaussian].")
  parser.add_argument("--kill-plateaus", default=False, action='store_true', help="Quit training if the model plateaus in the first 10 epochs.")

  # number of independent runs
  parser.add_argument("--runs", type=int, default=1, help="Number of independent runs [default = 1]. Ignored when using KD loss.")
  parser.add_argument("--start-run", type=int, default=0, help="Used to resume training, to skip already completed runs.")
  # datasets
  parser.add_argument("--dataset", type=str, default='cifar10', help="Available datasets: mnist, cifar10.")
  parser.add_argument("--dataset-path", type=str, default='./data', help="Path where datasets are stored.")
  # noisy labels
  parser.add_argument("--noisy", type=float, default=0, help="Percentage of corrupted labels [default = 0, max = 100].")

  # number of epochs for training each member of the ensemble
  parser.add_argument("--epochs", type=int, default=75, help="The number of epochs used for training [default = 75].")
  # minibatch size
  parser.add_argument("--batch-size", type=int, default=100, help="The minibatch size for training [default = 100].")
  # training criteria (scoring function)
  parser.add_argument("--loss", type=str, default='softmax', help="Supported loss functions: softmax, KD].")
  # the optimizer used to train each base classifier
  parser.add_argument("--optimizer", type=str, default='sgd', help="Supported optimizers: sgd, adam [default = sgd].")
  
  # base learning rate for SGD training
  parser.add_argument("--lr", type=float, default=0.1, help="The base learning rate for SGD optimization [default = 0.1].")
  # sgd step size
  parser.add_argument("--lr-step", type=int, default=20, help="The step size (# iterations) of the learning rate decay [default = 20].")
  # learning rate decay factor
  parser.add_argument("--lr-decay", type=float, default=0.1, help="The decay factor of the learning rate decay [default = 0.1].")
  # weight decay
  parser.add_argument("--weight-decay", type=float, default=5e-4, help="The weight decay coefficient [default = 0.0005 ].")
  # momentum for sgd
  parser.add_argument("--momentum", type=float, default=0.9, help="The momentum coefficient for SGD [default = 0.9].")
  
  # path where to store/load models
  parser.add_argument("--path", type=str, default=default_path, help="The dirname where to store/load models [default = './models'].")
  # snapshot frequency
  parser.add_argument("--snapshot-every", type=int, default=0, help="Snapshot the model state every E epochs [default = 0].")
  # path to a model snapshot, used to continue training
  parser.add_argument("--resume-from", type=str, default='', help="Path to a model snapshot [default = None]. For KD loss, this options specifies the teacher model file.")
  
  # knowledge distillation options
  parser.add_argument("--temperature", type=float, default=22., help="The softmax calibration factor [default = 22.0].")
  parser.add_argument("--alpha", type=float, default=0.7, help="Balance between KL and CrossEntropy terms in the KD loss [default = 0.7].")
  
  # log file
  parser.add_argument("--log", type=str, default='train.log', help="Logfile name [default = 'train.log'].")

  # parse the command line
  user_options = parser.parse_args()
  
  if user_options.arch is None:
    if user_options.loss == 'KD':
      print("Supported teacher architectures:")
      print(models.__teachers__)
      print("Supported student architectures:")
      print(models.__students__)
    else:
      print("Supported architectures:")
      print(models.__all__)
    sys.exit(0)
    
  path = user_options.path
  logdir = './log'
  logdir = os.path.join(
              os.path.join(logdir,user_options.dataset),
              user_options.arch
           )
  if user_options.noisy > 0:
    noisestr = 'noise_' + str(int(user_options.noisy * 100))
    logdir = os.path.join(logdir, noisestr)
  if not os.path.exists(path):
    os.mkdir(path)
  if not os.path.exists(logdir):
    os.makedirs(logdir)
  logfile = os.path.join(logdir, user_options.log)
  
  # init logging
  logger = logging.getLogger('train')
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
    
  logger.info('Using {} as root model directory.'.format(path))
  
  main(user_options)
