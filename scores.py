# -*- coding: utf-8 -*-

import torch
import sys

"""Validation accuracy
"""

def loss_plateaus(start_loss, current_loss, tol = 2.):
  """Check whether the loss plateaus. Useful when running HP search
  """
  with torch.no_grad():
    if current_loss == 0. or (current_loss != current_loss): # check for convergence or NaNs
      return True
    elif start_loss / current_loss < tol:
      return True
    else:
      return False
  
def test(net, test_loader, criterion, device):
  """Inference with net on the given test set
  """
  correct_cnt, avg_loss = 0, 0
  total_cnt = 0
  
  net = net.to(device=device)
  net = net.eval()

  with torch.no_grad():
    for batch_idx, (x, target) in enumerate(test_loader):
      x, target = x.to(device=device), target.to(device=device)
      out = net(x)
      loss = criterion(out, target)
      _, pred_label = torch.max(out.data, 1)
      total_cnt += x.data.size()[0]
      correct_cnt += (pred_label == target.data).sum()
      # smooth average
      avg_loss = avg_loss * 0.99 + loss.item() * 0.01
    accuracy = correct_cnt.item() * 1.0/total_cnt

  return avg_loss, accuracy
