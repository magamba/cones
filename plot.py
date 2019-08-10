#! /usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import numpy as np
import json
import matplotlib.pyplot as plt

"""
  discrete aggregators
"""

def pairwise_freq(nesting_dict):
  """
  The returned plot data has the form:
      plot_data:   
        { "epoch_id" = [ 
            { "pairs" = [
                 np.array(4, 2), ] }, ]
        } where each array contains the mean and std of the frequency count of each
          discrete feature, normalized by the corresponding number of filters per pair,
          averaged over runs
  """  
  plot_data = {}
  epoch = next(iter(nesting_dict))
  nruns = len(nesting_dict[str(epoch)])
  
  run = 0
  for epoch in nesting_dict:
    plot_data[str(epoch)] = []
    for block in nesting_dict[str(epoch)][str(run)]["blocks"]:
      block_dict = { "pairs" : [] }
      for pair in block["pairs"]:
        pair_freq = np.zeros((nruns,4), dtype='f')
        for kernel in pair["filters"]:
          pair_freq[run, :] += kernel
        if nruns == 1:
          mean = np.expand_dims(np.mean(pair_freq, 0), 1)
          std = np.expand_dims(np.std(pair_freq, 0), 1)
          pair_freq = np.concatenate((mean, std), 1)
        block_dict["pairs"].append(pair_freq)
      plot_data[str(epoch)].append(block_dict)        
  
  for run in range(1, nruns):
    for epoch in nesting_dict:
      for block_id, block in enumerate(nesting_dict[str(epoch)][str(run)]["blocks"]):
        block_dict = plot_data[str(epoch)][block_id]
        for pair_id, pair in enumerate(block["pairs"]):
          pair_freq = block_dict["pairs"][pair_id]
          for kernel in pair["filters"]:
            pair_freq[run, :] += kernel
          if run == nruns -1:
            # mean and std over runs
            mean = np.expand_dims(np.mean(pair_freq, 0), 1)
            std = np.expand_dims(np.std(pair_freq, 0), 1)
            pair_freq = np.concatenate((mean, std), 1)
            block_dict["pairs"][pair_id] = pair_freq
  return plot_data

def blockwise_freq(nesting_dict):
  """
  The returned plot data has the form:
      plot_data:   
        { "epoch_id" = [ 
             np.array(4, 2), ]
        } where each array contains the mean and std of the frequency count of each
          discrete feature, normalized by the corresponding number of filters per block,
          averaged over runs
  """
  plot_data = {}
  epoch = next(iter(nesting_dict))
  nruns = len(nesting_dict[str(epoch)])
  
  run = 0
  for epoch in nesting_dict:
    plot_data[str(epoch)] = []
    for block in nesting_dict[str(epoch)][str(run)]["blocks"]:
      block_freq = np.zeros((nruns,4), dtype='f')
      for pair in block["pairs"]:
        for kernel in pair["filters"]:
          block_freq[run, :] += kernel
      if nruns == 1:
        mean = np.expand_dims(np.mean(block_freq, 0), 1)
        std = np.expand_dims(np.std(block_freq, 0), 1)
        block_freq = np.concatenate((mean, std), 1)
      plot_data[str(epoch)].append(block_freq)
  
  for run in range(1, nruns):
    for epoch in nesting_dict:
      for block_id, block in enumerate(nesting_dict[str(epoch)][str(run)]["blocks"]):
        block_freq = plot_data[str(epoch)][block_id]
        for pair_id, pair in enumerate(block["pairs"]):
          for kernel in pair["filters"]:
            block_freq[run, :] += kernel
        if run == nruns -1:
          # mean and std over runs
          mean = np.expand_dims(np.mean(block_freq, 0), 1)
          std = np.expand_dims(np.std(block_freq, 0), 1)
          block_freq = np.concatenate((mean, std), 1)
          plot_data[str(epoch)][block_id] = block_freq
  return plot_data
  

def netwise_freq(nesting_dict):
  """
  The returned plot data has the form:
      plot_data:   
        { "epoch_id" = np.array(4, 2) }
          where each array contains the mean and std of the frequency count of each
          discrete feature, normalized by total number of filters in the network,
          averaged over runs
  """
  plot_data = {}
  epoch = next(iter(nesting_dict))
  nruns = len(nesting_dict[str(epoch)])
  
  for epoch in nesting_dict:
    plot_data[str(epoch)] = np.zeros((nruns,4), dtype='f')
 
  for run in range(nruns):
    for epoch in nesting_dict:
      for block_id, block in enumerate(nesting_dict[str(epoch)][str(run)]["blocks"]):
        for pair_id, pair in enumerate(block["pairs"]):
          for kernel in pair["filters"]:
            plot_data[str(epoch)][run, :] += kernel
      if run == nruns -1:
        # mean and std over runs
        mean = np.mean(plot_data[str(epoch)], 0)
        mean = np.expand_dims(mean, 1)
        std = np.std(plot_data[str(epoch)], 0)
        std = np.expand_dims(std, 1)
        nesting_freq = np.concatenate((mean, std), 1)
        plot_data[str(epoch)] = nesting_freq
  return plot_data

"""
 continuous aggregators
"""

"""
  Distance normalization:
    - for each run, distance should be normalized by the maximum value over
      all runs
    - given nesting_freq:
      - for each run, compute max distance:
        - netwise
        - blockwise
        - pairwise
        
  Aggregating continuous stats:
    - make array of values per run
    - normalize distance for each run
    - compute mean and std per point
    - use std to compute gaussian kernels
    
  Problem: kernel bandwidth
    - each filter is observed over nruns trials
    - each observation is unnormalized
    
    - for each of pairwise, netwise and blockwise
      for each run
        filterwise distance should be normalized by max[pair|block|net]
      mean values over runs are computed
      std over runs is used to estimate the kernel bandwidth
      
   TODO:
     1. visit continuous stats to get normalizing constants per run
     2. array of values [nruns, numfilters, 3]
     3. nesting stats [numfilters, 3, 2]
     
     4. for each epoch, for each blocktype, for each measure, plot
     5. store plots in plots/dataset/arch/[net|block|pair]_[disc|cont]_epoch.png
"""

def normalize_distance(distance_array, max_dist=None):
  """Normalize distance by its maximum value
  """
  # compute maximum
  if max_dist is None:
    max_dist = np.max(distance_array)
  
  # divide each element by maximum
  if max_dist > 0:
    distance_array = distance_array / max_dist
  # return normalized list
  return distance_array

def pairwise_dist(nesting_dict):
  """
  The returned plot data has the form:
      plot_data:   
        { "epoch_id" = [ 
            { "pairs" = [
                 np.array(numfilters, 3, 2), ] }, ]
        } where each array contains the mean observed values over nruns of the continouous
          features, with distance normalized by the largest observed value over all pairs
  """  
  plot_data = {}
  epoch = next(iter(nesting_dict))
  nruns = len(nesting_dict[str(epoch)])
  
  run = 0
  for epoch in nesting_dict:
    plot_data[str(epoch)] = []
    for block in nesting_dict[str(epoch)][str(run)]["blocks"]:
      block_dict = { "pairs" : [] }
      for pair in block["pairs"]:
        pair_dist = np.zeros((nruns,len(pair["filters"]),3), dtype='f')
        for kernel_id, kernel in enumerate(pair["filters"]):
          pair_dist[run, kernel_id, :] = kernel
        if nruns == 1:
          # normalize distance
          pair_dist[0, :, 0] = normalize_distance(pair_dist[0, :, 0])
          # mean and std over runs
          mean = np.expand_dims(np.mean(pair_dist, 0), 2)
          std = np.expand_dims(np.std(pair_dist, 0), 2)
          pair_dist = np.concatenate((mean, std), 2)
        block_dict["pairs"].append(pair_dist)
      plot_data[str(epoch)].append(block_dict)
  
  for run in range(1, nruns):
    for epoch in nesting_dict:
      for block_id, block in enumerate(nesting_dict[str(epoch)][str(run)]["blocks"]):
        block_dict = plot_data[str(epoch)][block_id]
        for pair_id, pair in enumerate(block["pairs"]):
          pair_dist = block_dict["pairs"][pair_id]
          for kernel_id, kernel in enumerate(pair["filters"]):
            pair_dist[run, kernel_id, :] = kernel
          if run == nruns -1:
            # normalize distance
            max_dist = np.max(np.max(pair_dist[:,:,0], 1), 0)
            for run_id in range(pair_dist.shape[0]):
              pair_dist[run_id, :, 0] = normalize_distance(pair_dist[run_id, :, 0], max_dist)
            # mean and std over runs
            mean = np.expand_dims(np.mean(pair_dist, 0), 2)
            std = np.expand_dims(np.std(pair_dist, 0), 2)
            pair_dist = np.concatenate((mean, std), 2)
            block_dict["pairs"][pair_id] = pair_dist
  return plot_data

def blockwise_dist(nesting_dict, filter_count):
  """
  The returned plot data has the form:
      plot_data:   
        { "epoch_id" = [ 
                 np.array(numfilters, 3, 2), ]
        } where each array contains the mean observed values over nruns of the continouous
          features, with distance normalized by the largest observed value within each block
  """  
  plot_data = {}
  epoch = next(iter(nesting_dict))
  nruns = len(nesting_dict[str(epoch)])
  
  run = 0
  for epoch in nesting_dict:
    plot_data[str(epoch)] = []
    for block_id, block in enumerate(nesting_dict[str(epoch)][str(run)]["blocks"]):
      nfilters = filter_count["blocks"][block_id]["num_filters"]
      block_dist = np.zeros((nruns,nfilters,3), dtype='f')
      fcounter = 0
      for pair in block["pairs"]:
        for kernel_id, kernel in enumerate(pair["filters"]):
          block_dist[run, fcounter, :] = kernel
          fcounter += 1
      if nruns == 1:
        # normalize distance
        block_dist[0, :, 0] = normalize_distance(block_dist[0, :, 0])
        # mean and std over runs
        mean = np.expand_dims(np.mean(block_dist, 0), 2)
        std = np.expand_dims(np.std(block_dist, 0), 2)
        block_dist = np.concatenate((mean, std), 2)
      plot_data[str(epoch)].append(block_dist)
  
  for run in range(1, nruns):
    for epoch in nesting_dict:
      for block_id, block in enumerate(nesting_dict[str(epoch)][str(run)]["blocks"]):
        block_dist = plot_data[str(epoch)][block_id]
        fcounter = 0
        for pair_id, pair in enumerate(block["pairs"]):
          for kernel_id, kernel in enumerate(pair["filters"]):
            block_dist[run, fcounter, :] = kernel
            fcounter += 1
        if run == nruns -1:
          # normalize distance
          max_dist = np.max(np.max(block_dist[:, :, 0], 1), 0)
          for run_id in range(nruns):
            block_dist[run_id, :, 0] = normalize_distance(block_dist[run_id, :, 0], max_dist)
          # mean and std over runs
          mean = np.expand_dims(np.mean(block_dist, 0), 2)
          std = np.expand_dims(np.std(block_dist, 0), 2)
          block_dist = np.concatenate((mean, std), 2)
        plot_data[str(epoch)][block_id] = block_dist
  return plot_data

def netwise_dist(nesting_dict, filter_count):
  """
  The returned plot data has the form:
      plot_data:   
        { "epoch_id" = np.array(numfilters, 3, 2) }
        where each array contains the mean observed values over nruns of the continouous
        features, with distance normalized by the largest observed value within each block
  """  
  plot_data = {}
  epoch = next(iter(nesting_dict))
  nruns = len(nesting_dict[str(epoch)])
  
  run = 0
  for epoch in nesting_dict:
    plot_data[str(epoch)] = ""
    nfilters = filter_count["num_filters"]
    net_dist = np.zeros((nruns,nfilters,3), dtype='f')
    fcounter = 0
    for block_id, block in enumerate(nesting_dict[str(epoch)][str(run)]["blocks"]):
      for pair in block["pairs"]:
        for kernel_id, kernel in enumerate(pair["filters"]):
          net_dist[run, fcounter, :] = kernel
          fcounter += 1
      if nruns == 1:
        # normalize distance
        net_dist[0, :, 0] = normalize_distance(net_dist[0, :, 0])
        # mean and std over runs
        mean = np.expand_dims(np.mean(net_dist, 0), 2)
        std = np.expand_dims(np.std(net_dist, 0), 2)
        net_dist = np.concatenate((mean, std), 2)
    plot_data[str(epoch)] = net_dist
  
  for run in range(1, nruns):
    for epoch in nesting_dict:
      net_dist = plot_data[str(epoch)]
      fcounter = 0
      for block_id, block in enumerate(nesting_dict[str(epoch)][str(run)]["blocks"]):
        for pair_id, pair in enumerate(block["pairs"]):
          for kernel_id, kernel in enumerate(pair["filters"]):
            net_dist[run, fcounter, :] = kernel
            fcounter += 1
      if run == nruns -1:
        # normalize distance
        max_dist = np.max(np.max(net_dist[:, :, 0], 1), 0)
        for run_id in range(nruns):
          net_dist[run_id, :, 0] = normalize_distance(net_dist[run_id, :, 0], max_dist)
        # mean and std over runs
        mean = np.expand_dims(np.mean(net_dist, 0), 2)
        std = np.expand_dims(np.std(net_dist, 0), 2)
        net_dist = np.concatenate((mean, std), 2)
      plot_data[str(epoch)] = net_dist
  return plot_data
  
  
"""
  Generators
"""

def load_json(file_handler):
  """Generator returning a dictionary of results
     for each line in file_handles
  """
  for line in file_handler:
    line = line.strip()
    print('Loading {}'.format(line))
    with open(line, 'rb') as l:
      results_dict = json.load(l, encoding='utf-8')
    yield results_dict
    
def next_pair_disc(plot_data, epoch):
  """Yields pairwise discrete distribution for the specified epoch
  """
  for block in plot_data[str(epoch)]:
    for pair_freq in block["pairs"]:
      yield pair_freq
      
def next_pair_cont(plot_data, epoch):
  """Yields pairwise continuous observed values for the specified epoch
  """
  for block in plot_data[str(epoch)]:
    for pair_dist in block["pairs"]:
      yield pair_dist

def next_block_disc(plot_data, epoch):
  """Yields blockwise discrete distribution for the specified epoch
  """
  for block_freq in plot_data[str(epoch)]:
    yield block_freq
      
def next_block_cont(plot_data, epoch):
  """Yields blockwise continuous observed values for the specified epoch
  """
  for block_dist in plot_data[str(epoch)]:
    yield block_dist
    
"""
  Results preprocessing
"""
    
def get_filter_count(results):
  """
  Normalizing constants to estimate the distribution
  of the computed measure.
  
  Given any dictionary of results implicitly describing
  a network architecture, the method returns a dictionary:
  
   filter_count = {
         "num_filters" : num_filters, # network count
         "blocks" : [
            {
              "num_filters" : num_filters # block count
              "pairs: [
                  num_filters, # pair count
              ]
            },
         ]
    }
  """
  filter_count = { "blocks" : [] }
  network_count = 0
  
  for block_id, block in enumerate(results["blocks"]):
    pairs_dict = { "pairs" : [] }
    block_count = 0
    
    for pair_id, pair in enumerate(results["blocks"][block_id]["layer_pairs"]):
      # check number of output filters
      pair_count = len(results["blocks"][block_id]["layer_pairs"][pair_id])
      pairs_dict["pairs"].append(pair_count)
      block_count += pair_count
      
    pairs_dict["num_filters"] = block_count
    filter_count["blocks"].append(pairs_dict)
    network_count += block_count
  filter_count["num_filters"] = network_count
  return filter_count
    
def collect_cont_stats(results_dict, nesting_freq=None, rel_runs=0):
  """ Parse the result dictionary and collect the unnormalized frequency of
     the discrete nesting stats for each epoch
      
      If plot_data is not None, results are inserted into nesting_freq
      otherwise, a new dictionary is created and returned.
      
      The returned plot data has the form:
      nesting_freq:   
        { "epoch_id" = {
            "run_id" = {
              "blocks" : [
                {
                  "pairs" = [
                    {
                      "filters" : [
                          np.array(3) ] # array[i] is the ith feature stored
                    },
                  ]
                }, 
              ] 
            } 
          } 
        } where each array contains the raw frequency count of each
          discrete feature, unnormalized
  """
  
  if rel_runs > 0:
    run = str( int(results_dict["run"]) % rel_runs )
  else:
    run = str(results_dict["run"])
  epoch = str(results_dict["epoch"])
  
  if nesting_freq is None:
    nesting_freq = {}
    
  try:
    run_dict = nesting_freq[epoch]
  except KeyError:
    nesting_freq[epoch] = {}
  
  nesting_freq[epoch][run] = { "blocks" : [] }
  
  for block_id, block in enumerate(results_dict["blocks"]):
    block_dict = { "pairs" : [] }
    for pair_id, pair in enumerate(results_dict["blocks"][block_id]["layer_pairs"]):
      pair_dict = { "filters" : [] }
      for kernel_id, kernel in enumerate(results_dict["blocks"][block_id]["layer_pairs"][pair_id]):
        # state_freq[i] is the frequency of state i
        state_freq = np.zeros(3, dtype='f')
        state_freq[0] = results_dict["blocks"][block_id]["layer_pairs"][pair_id][kernel_id]["distance"]
        state_freq[1] = results_dict["blocks"][block_id]["layer_pairs"][pair_id][kernel_id]["opening"]
        state_freq[2] = results_dict["blocks"][block_id]["layer_pairs"][pair_id][kernel_id]["rotation"]
        pair_dict["filters"].append(state_freq)
      block_dict["pairs"].append(pair_dict)
    nesting_freq[epoch][run]["blocks"].append(block_dict)
  return nesting_freq
    
def collect_discrete_stats(results_dict, nesting_freq=None, rel_runs=0):
  """ Parse the result dictionary and collect the unnormalized frequency of
     the discrete nesting stats for each epoch
      
      If plot_data is not None, results are inserted into nesting_freq
      otherwise, a new dictionary is created and returned.
      
      The returned plot data has the form:
      nesting_freq:   
        { "epoch_id" = {
            "run_id" = {
              "blocks" : [
                {
                  "pairs" = [
                    {
                      "filters" : [
                          np.array(4) ] 
                    },
                  ]
                }, 
              ] 
            } 
          } 
        } where each array contains the raw frequency count of each
          discrete feature, unnormalized
  """
  if rel_runs > 0:
    run = str( int(results_dict["run"]) % rel_runs )
  else:
    run = str(results_dict["run"])
  epoch = str(results_dict["epoch"])
  
  if nesting_freq is None:
    nesting_freq = {}
    
  try:
    run_dict = nesting_freq[epoch]
  except KeyError:
    nesting_freq[epoch] = {}
  
  nesting_freq[epoch][run] = { "blocks" : [] }
  
  for block_id, block in enumerate(results_dict["blocks"]):
    block_dict = { "pairs" : [] }
    for pair_id, pair in enumerate(results_dict["blocks"][block_id]["layer_pairs"]):
      pair_dict = { "filters" : [] }
      for kernel_id, kernel in enumerate(results_dict["blocks"][block_id]["layer_pairs"][pair_id]):
        # state_freq[i] is the frequency of state i
        state_freq = np.zeros(4, dtype='f')
        state_freq[results_dict["blocks"][block_id]["layer_pairs"][pair_id][kernel_id]["binary_nesting"]] += 1
        pair_dict["filters"].append(state_freq)
      block_dict["pairs"].append(pair_dict)
    nesting_freq[epoch][run]["blocks"].append(block_dict)
  return nesting_freq
  
def arch_name(name_str):
  split_name = name_str.split('_')
  name = ''
  for index, s in enumerate(split_name[0:len(split_name)-1]):
    name = name + s
    if index < len(split_name) -2:
      name = name + '_'
  return name
  
"""
  Histograms
"""

def plot_hist_net_epochs(plot_data, filter_count=None, path=None):
  """Plot histogram of frequencies for all epochs 
     and (optionally) saves the plot to file if
     a path is specified
     If a filter count dictionary is specified, the histogram
     absolute frequency and error are normalized before plotting
  """
  import constants
  labels = constants.state_names
  
  freqs = []
  stdevs = []
  
  if filter_count is not None:
    nfilters = float(filter_count["num_filters"])
  else:
    nfilters = 1.
  
  for epoch in plot_data:
    freq = (plot_data[epoch][:,0] / nfilters ).tolist()
    stdev = (plot_data[epoch][:,1] / nfilters ).tolist()
    freqs.append(freq)
    stdevs.append(stdev)
  
  freqs = np.array(freqs).T
  stdevs = np.array(stdevs).T
    
  fig, ax = plt.subplots(figsize=(20, 7))
    
  gap = .8 / len(freqs)
  for i, row in enumerate(freqs):
    xpos = np.arange(len(row))
    ax.bar(xpos + i * gap, row, yerr=stdevs[i], width=gap, alpha=0.5, ecolor='black', capsize=10, label=labels[i])
    
  ax.set_ylabel('Netwise frequency')
  ax.set_xlabel('Epochs')
  ax.set_ylim(0,1)
  ax.set_xticks(xpos)
  ax.set_xticklabels(xpos * 10)
  ax.legend(loc='upper right')
  
  plt.tight_layout()
  
  if path is not None:
    filename = 'net_disc_global.png'
    plt.savefig(os.path.join(path, filename), bbox_inches='tight', dpi=300)
  else:
    plt.show()
    
  plt.close(fig)

def plot_hist_net(plot_data, epoch, filter_count=None, path=None):
  """Plot histogram of frequencies for the specified
     epoch and (optionally) saves the plot to file if
     a path is specified
     If a filter count dictionary is specified, the histogram
     absolute frequency and error are normalized before plotting
  """
  import constants
  
  labels = constants.state_names
  xpos = np.arange(len(labels))
  
  freq = plot_data[str(epoch)][:, 0]
  stdev = plot_data[str(epoch)][:,1]
  
  if filter_count is not None:
    nfilters = float(filter_count["num_filters"])
    freq = freq / nfilters
    stdev = stdev / nfilters
  
  fig, ax = plt.subplots()
  ax.bar(xpos, freq, yerr=stdev, align='center', alpha=0.5, ecolor='black', capsize=10)
  ax.set_ylabel('Netwise frequency')
  ax.set_ylim(0,1)
  ax.set_xticks(xpos)
  ax.set_xticklabels(labels)
  ax.set_title('Epoch ' + str(epoch))
  ax.yaxis.grid(True)

  plt.tight_layout()
  
  if path is not None:
    filename = 'net_disc_' + str(epoch) + '.png'
    plt.savefig(os.path.join(path, filename))
  else:
    plt.show()
    
  plt.close(fig)

def plot_hist_block(plot_data, epoch, filter_count, normalize=True, path=None):
  """Plot histogram of frequencies for the specified
     epoch and (optionally) saves the plot to file if
     a path is specified
     If normalize is True, the histogram
     absolute frequency and error are normalized before plotting
  """
  import constants
  
  labels = constants.state_names
  xpos = np.arange(len(labels))
  
  block_id = 0
  num_blocks = len(filter_count["blocks"])
  fig, ax = plt.subplots(1, num_blocks)
  
  for block_data in next_block_disc(plot_data, epoch):
    freq = block_data[:, 0]
    stdev = block_data[:,1]
  
    if normalize:
      nfilters = float(filter_count["blocks"][block_id]["num_filters"])
      freq = freq / nfilters
      stdev = stdev / nfilters
    
    ax[block_id].bar(xpos, freq, yerr=stdev, align='center', alpha=0.5, ecolor='black', capsize=10)
    ax[block_id].set_ylabel('Blockwise frequency')
    ax[block_id].set_ylim(0,1)
    ax[block_id].set_xticks(xpos)
    ax[block_id].set_xticklabels(labels)
    ax[block_id].set_title('Block ' + str(block_id +1) + ': epoch ' + str(epoch))
    ax[block_id].yaxis.grid(True)
    plt.setp(ax[block_id].get_xticklabels(), rotation=30, horizontalalignment='right')
    block_id += 1

  plt.tight_layout()
  
  if path is not None:
    filename = 'block_disc_' + str(epoch) + '.png'
    plt.savefig(os.path.join(path, filename))
  else:
    plt.show()
    
  plt.close(fig)

def plot_hist_pair(plot_data, epoch, filter_count, normalize=True, path=None):
  """Plot histogram of frequencies for the specified
     epoch and (optionally) saves the plot to file if
     a path is specified
     If normalize is True, the histogram
     absolute frequency and error are normalized before plotting
  """
  import constants
  
  labels = constants.state_names
  xpos = np.arange(len(labels))
  
  num_blocks = len(filter_count["blocks"])
  
  pair_per_block = []
  max_numpairs = 0
  for block in filter_count["blocks"]:
    num_pairs = len(block["pairs"])
    if num_pairs > max_numpairs:
      max_pairs = num_pairs
    pair_per_block.append(num_pairs)

  fig, ax = plt.subplots(num_blocks, max_pairs, squeeze=False)
  pair_gen = next_pair_disc(plot_data, epoch)
  
  for block_id in range(num_blocks):
    for pair_id in range(pair_per_block[block_id]):
      pair_data = next(pair_gen)
      freq = pair_data[:, 0]
      stdev = pair_data[:,1]
    
      if normalize:
        nfilters = float(filter_count["blocks"][block_id]["pairs"][pair_id])
        freq = freq / nfilters
        stdev = stdev / nfilters
      
      ax[block_id, pair_id].bar(xpos, freq, yerr=stdev, align='center', alpha=0.5, ecolor='black', capsize=10)
      ax[block_id, pair_id].set_ylabel('Pairwise frequency')
      ax[block_id, pair_id].set_ylim(0,1)
      ax[block_id, pair_id].set_xticks(xpos)
      ax[block_id, pair_id].set_xticklabels(labels)
      ax[block_id, pair_id].set_title('Block ' + str(block_id +1) + ', pair ' + str(pair_id +1) + ': epoch ' + str(epoch))
      ax[block_id, pair_id].yaxis.grid(True)

  plt.tight_layout()
  
  if path is not None:
    filename = 'pair_disc_' + str(epoch) + '.png'
    plt.savefig(os.path.join(path, filename))
  else:
    plt.show()

  plt.close(fig)

"""
  KDE
"""

def gaussian_kernel(x, val_array, bandwidth_array):
  """Compute scalar Gaussian kernels at x,
     each centered the corresponding item of val_array, with width bandwidth_array
  """
  return (1. / np.sqrt(2.* np.pi * bandwidth_array**2)) * np.exp(- (x - val_array) * (x - val_array) / (2. * bandwidth_array * bandwidth_array))

def compute_kde(x_list, val_array, bandwidth_array):
  """Estimate the density of val over the linspace x
     using a per-point kernel bandwidth
  """
  kde_array = np.array([])
  for x in x_list:
    kernel = gaussian_kernel(x, val_array, bandwidth_array)
    if len(kde_array) == 0:
      kde_array = kernel
    else:
      kde_array = np.vstack((kde_array, kernel))
  return np.sum(kde_array, 1) / len(val_array)
  
def plot_kde_pair(plot_data, bandwidth, path=None, rotations=False):
  """Estimate the distribution of each continuous statistic
     and (optionally) save the plot to file if a path is specified
  """
  
  num_pairs = 0
  for pair in next_pair_cont(plot_data, 9):
    num_pairs += 1
    
  if rotations:
    fig, ax = plt.subplots(num_pairs, 3, squeeze=False)
  else:
    fig, ax = plt.subplots(num_pairs, 2, squeeze=False)
  xpos = np.linspace(0,0.4,100)
  
  for epoch in plot_data:
    for pair_id, pair_data in enumerate(next_pair_cont(plot_data, epoch)):
  
      dist_mean = pair_data[:,0,0]
      open_mean = pair_data[:,1,0]
      rot_mean = pair_data[:,2,0]
    
      dist_std = pair_data[:,0,1]
      open_std = pair_data[:,1,1]
      rot_std = pair_data[:,2,1]
      
      bandwidth_array = bandwidth * np.ones(dist_mean.shape[0], dtype='f')
      dist_bandwidth = np.sqrt(bandwidth_array **2 + dist_std **2)
      open_bandwidth = np.sqrt(bandwidth_array **2 + open_std **2)
      rot_bandwidth = np.sqrt(bandwidth_array **2 + rot_std **2)
      
      # compute KDE
      kde_dist = compute_kde(xpos, dist_mean, dist_bandwidth) 
      kde_open = compute_kde(xpos, open_mean, open_bandwidth)
      kde_rot = compute_kde(xpos, rot_mean, rot_bandwidth)
      
      ax[pair_id, 0].plot(xpos, kde_dist, label='e '+str(epoch))
      ax[pair_id, 0].plot(dist_mean, np.full_like(dist_mean, -0.1), '|k', markeredgewidth=1)
      ax[pair_id, 0].set_xlabel('distance')
      ax[pair_id, 0].set_title('Pair ' + str(pair_id +1))
      
      ax[pair_id, 1].plot(xpos, kde_open, label='e '+str(epoch))
      ax[pair_id, 1].plot(open_mean, np.full_like(open_mean, -0.1), '|k', markeredgewidth=1)
      ax[pair_id, 1].set_xlabel('opening')
      ax[pair_id, 1].set_title('Pair ' + str(pair_id +1))
      
      if rotations:
        ax[pair_id, 2].plot(xpos, kde_rot, label='e '+str(epoch))
        ax[pair_id, 2].plot(rot_mean, np.full_like(rot_mean, -0.1), '|k', markeredgewidth=1)
        ax[pair_id, 2].set_xlabel('rotation')
        ax[pair_id, 2].set_title('Pair ' + str(pair_id +1))
    
  for axs in ax.ravel():
    axs.set_ylabel('pdf')
    axs.legend(loc='upper right')
  
  plt.tight_layout()
  
  if path is not None:
    filename = 'pair_cont.png'
    plt.savefig(os.path.join(path, filename))
  else:
    plt.show()
    
  plt.close(fig)
  
def plot_kde_block(plot_data, bandwidth, path=None, rotations=False):
  """Estimate the distribution of each continuous statistic
     and (optionally) save the plot to file if a path is specified
  """
  
  num_blocks = 0
  for block in next_block_cont(plot_data, 9):
    num_blocks += 1
    
  if rotations:
    fig, ax = plt.subplots(num_blocks, 3, squeeze=False)
  else:
    fig, ax = plt.subplots(num_blocks, 2, squeeze=False, figsize=(10, 10))
  xpos = np.linspace(0,0.4,100)
  
  for epoch in plot_data:
    for block_id, block_data in enumerate(next_block_cont(plot_data, epoch)):
  
      dist_mean = block_data[:,0,0]
      open_mean = block_data[:,1,0]
      rot_mean = block_data[:,2,0]
    
      dist_std = block_data[:,0,1]
      open_std = block_data[:,1,1]
      rot_std = block_data[:,2,1]
      
      bandwidth_array = bandwidth * np.ones(dist_mean.shape[0], dtype='f')
      dist_bandwidth = np.sqrt(bandwidth_array **2 + dist_std **2)
      open_bandwidth = np.sqrt(bandwidth_array **2 + open_std **2)
      rot_bandwidth = np.sqrt(bandwidth_array **2 + rot_std **2)
      
      # compute KDE
      kde_dist = compute_kde(xpos, dist_mean, dist_bandwidth) 
      kde_open = compute_kde(xpos, open_mean, open_bandwidth)
      kde_rot = compute_kde(xpos, rot_mean, rot_bandwidth)
      
      ax[block_id, 0].plot(xpos, kde_dist, label='e '+str(epoch))
      ax[block_id, 0].plot(dist_mean, np.full_like(dist_mean, -0.1), '|k', markeredgewidth=1)
      ax[block_id, 0].set_title('Block ' + str(block_id +1))
      
      ax[block_id, 1].plot(xpos, kde_open, label='e '+str(epoch))
      ax[block_id, 1].plot(open_mean, np.full_like(open_mean, -0.1), '|k', markeredgewidth=1)
      ax[block_id, 1].set_title('Block ' + str(block_id +1))
      
      if rotations:
        ax[block_id, 2].plot(xpos, kde_rot, label='e '+str(epoch))
        ax[block_id, 2].plot(rot_mean, np.full_like(rot_mean, -0.1), '|k', markeredgewidth=1)
        ax[block_id, 2].set_title('Block ' + str(block_id +1))
  
  for axs in ax.ravel():
    axs.set_ylabel('pdf')
  
  ax[0, 0].legend(loc='upper right', bbox_to_anchor=(1,1))
  ax[0, 0].set_zorder(100)
  ax[0, 1].legend(loc='upper right', bbox_to_anchor=(1,1))
  ax[0, 1].set_zorder(100)
  
  ax[block_id, 0].set_xlabel('distance')
  ax[block_id, 1].set_xlabel('opening')
  
  if rotations:
    ax[0, 2].legend(loc='upper right', bbox_to_anchor=(1,1))
    ax[0, 2].set_zorder(100)
    ax[block_id, 2].set_xlabel('rotation')
  
  plt.tight_layout()
  
  if path is not None:
    filename = 'block_cont.png'
    plt.savefig(os.path.join(path, filename), bbox_inches='tight', dpi=300)
  else:
    plt.show()
    
  plt.close(fig)

def plot_kde_net(plot_data, bandwidth, path=None, rotations=False):
  """Estimate the distribution of each continuous statistic
     and (optionally) save the plot to file if a path is specified
  """
  
  if rotations:
    fig, ax = plt.subplots(1,3)
  else:
    fig, ax = plt.subplots(1,2, figsize=(10,10))
  xpos = np.linspace(0,0.4,100)
  
  for epoch in plot_data:
    dist_mean = plot_data[str(epoch)][:,0,0]
    open_mean = plot_data[str(epoch)][:,1,0]
    rot_mean = plot_data[str(epoch)][:,2,0]
  
    dist_std = plot_data[str(epoch)][:,0,1]
    open_std = plot_data[str(epoch)][:,1,1]
    rot_std = plot_data[str(epoch)][:,2,1]
    
    bandwidth = bandwidth * np.ones(dist_mean.shape[0], dtype='f')
    dist_bandwidth = np.sqrt(bandwidth **2 + dist_std **2)
    open_bandwidth = np.sqrt(bandwidth **2 + open_std **2)
    rot_bandwidth = np.sqrt(bandwidth **2 + rot_std **2)
    
    # compute KDE
    kde_dist = compute_kde(xpos, dist_mean, dist_bandwidth) 
    kde_open = compute_kde(xpos, open_mean, open_bandwidth)
    kde_rot = compute_kde(xpos, rot_mean, rot_bandwidth)
    
    ax[0].plot(xpos, kde_dist, label='e '+str(epoch))
    ax[0].plot(dist_mean, np.full_like(dist_mean, -0.1), '|k', markeredgewidth=1)
    ax[0].set_xlabel('distance')
    
    ax[1].plot(xpos, kde_open, label='e '+str(epoch))
    ax[1].plot(open_mean, np.full_like(open_mean, -0.1), '|k', markeredgewidth=1)
    ax[1].set_xlabel('opening')
    
    if rotations:
      ax[2].plot(xpos, kde_rot, label='e '+str(epoch))
      ax[2].plot(rot_mean, np.full_like(rot_mean, -0.1), '|k', markeredgewidth=1)
      ax[2].set_xlabel('rotation')
    
  for axs in ax:
    axs.set_ylabel('pdf')
    axs.legend(loc='upper right', bbox_to_anchor=(1,1))
  
  plt.tight_layout()
  
  if path is not None:
    filename = 'net_cont.png'
    plt.savefig(os.path.join(path, filename), bbox_inches='tight', dpi=300)
  else:
    plt.show()
    
  plt.close(fig)
  
"""
  Main
"""

def main(args):
  """Generate plots from the specified list
   result dictionaries
  """
  
  if not os.path.exists(args.load_from):
    raise FileNotFoundError('File {} does not exist.'.format(filename))
  
  with open(args.load_from, 'r') as fp:
    nesting_stats_disc = None # dictionary of discrete nesting stats
    nesting_stats_cont = None # dictionary of continuous nesting stats
    
    for results in load_json(fp): # iterate through stored results
      if args.discrete:
        nesting_stats_disc = collect_discrete_stats(results, nesting_stats_disc, args.runs)
    
      if args.continuous:
        nesting_stats_cont = collect_cont_stats(results, nesting_stats_cont, args.runs)
      
  name = arch_name(results["name"])
  arch = results["arch"]
  dataset = results["dataset"]
  
  # path to save plots
  if args.save:
    path = os.path.join(os.path.join(args.path, dataset), name)
    if not os.path.exists(path):
      os.makedirs(path)
  else:
    path = None  

  # normalizing constants
  filter_counts = get_filter_count(results)
  
  # compute aggregators
  if nesting_stats_disc is not None:
    num_epochs = len(nesting_stats_disc)
    num_runs = len(nesting_stats_disc[next(iter(nesting_stats_disc))])
  
    if args.pair:
      plot_freq_pair = pairwise_freq(nesting_stats_disc)
      # plot data and save to path
      for epoch in plot_freq_pair:
        plot_hist_pair(plot_freq_pair, epoch, filter_counts, path=path)
      
    if args.block:
      plot_freq_block = blockwise_freq(nesting_stats_disc)
      # plot data and save to path
      for epoch in plot_freq_block:
        plot_hist_block(plot_freq_block, epoch, filter_counts, path=path)
      
    if args.net:
      plot_freq_net = netwise_freq(nesting_stats_disc)
      # plot data and save to path
      if args.group:
        plot_hist_net_epochs(plot_freq_net, filter_counts, path=path)
      else:
        for epoch in plot_freq_net:
          plot_hist_net(plot_freq_net, epoch, filter_counts, path=path)
  
  if nesting_stats_cont is not None:
    num_epochs = len(nesting_stats_cont)
    num_runs = len(nesting_stats_cont[next(iter(nesting_stats_cont))])
    
    if args.pair:
      plot_dist_pair = pairwise_dist(nesting_stats_cont)
      # plot data and save to path
      plot_kde_pair(plot_dist_pair, args.bandwidth, path=path, rotations=args.rotations)
      
    if args.block:
      plot_dist_block = blockwise_dist(nesting_stats_cont, filter_counts)
      # plot data and save to path
      plot_kde_block(plot_dist_block, args.bandwidth, path=path, rotations=args.rotations)
      
    if args.net:
      plot_dist_net = netwise_dist(nesting_stats_cont, filter_counts)
      # plot data and save to path
      plot_kde_net(plot_dist_net, args.bandwidth, path=path, rotations=args.rotations)

if __name__ == '__main__':

  default_path = "./plots"

  # parse command line arguments
  import argparse
  parser = argparse.ArgumentParser()
  
  # path where to store/load models
  parser.add_argument("--path", type=str, default=default_path, help="The dirname where to store plots [default = './plots'].")
  
  # whether to save the plots to file
  parser.add_argument("--save", default=False, action='store_true', help="Whether to save the plots to file.")
  
  # number of relative runs
  parser.add_argument("--runs", default=0, help="Total number of runs, (optional). If the listed runs are not in the range [0, nruns), use this value to normalize the range.")
  
  # path to saved json results
  parser.add_argument("--load-from", type=str, default='', help="File with the list of saved results, one for each line.")
  
  # plot binary nesting stats
  parser.add_argument("--discrete", default=False, action='store_true', help="Whether to plot the discrete nesting distribution.")
  
  # plot full nesting stats
  parser.add_argument("--continuous", default=False, action='store_true', help="Whether to plot the continuous nesting distribution.")
  
  # bandwidth for KDE
  parser.add_argument("--bandwidth", type=float, default=1., help="Gaussian kernel bandwidth for KDE [default = 1].")
  
  # plot network-wise nesting
  parser.add_argument("--net", default=False, action='store_true', help="Whether to plot the network-wise nesting distribution.")
  
  # plot block nesting
  parser.add_argument("--block", default=False, action='store_true', help="Whether to plot the block-wise nesting distribution.")
  
  # plot layer_pair nesting
  parser.add_argument("--pair", default=False, action='store_true', help="Whether to plot the layer pair-wise nesting distribution.")
  
  # aggregate net histograms into a single plot
  parser.add_argument("--group", default=False, action='store_true', help="If specified with --net, the histograms of all epochs are aggregated in a single plot.")
  
  # plot layer_pair nesting
  parser.add_argument("--rotations", default=False, action='store_true', help="Whether to plot rotation statistics.")
    
  args = parser.parse_args()
  
  if not (args.discrete or args.continuous):
    print("Please specify whether continuous or discrete stats should be plotted.")
    sys.exit(0)
  
  if not os.path.exists(args.path):
    os.makedirs(args.path)

  main(args)
