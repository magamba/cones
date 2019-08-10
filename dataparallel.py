# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

class NamedDataParallel(nn.DataParallel):
  """ Extends data parallel by allowing access to
      model attributes
  """
 
  def __init__(self, module):
    super(NamedDataParallel, self).__init__(module)
    self.__name__ = module.__name__

