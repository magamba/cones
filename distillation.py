# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.nn import functional as F

"""
Knowledge distillation loss
"""

class KDLoss(nn.Module):

  def __init__(self, temp = 1., alpha = 0.):
    super(KDLoss, self).__init__()
    self.temp = temp
    if alpha >= 0. and alpha <= 1.:
      self.alpha = alpha
    else:
      raise ValueError('Alpha should be a float in [0,1].')
    
  def forward(self, x, labels, teacher_logits):
    """ Compute the knowledge distillation loss
        with parameter alpha and temperature temp
    """
    loss = nn.KLDivLoss()(F.log_softmax(x / self.temp, dim=1), 
                          F.softmax(teacher_logits / self.temp, dim=1)) * (self.alpha * self.temp * self.temp) + \
                          F.cross_entropy(x, labels) * (1. - self.alpha)
                          
    return loss
