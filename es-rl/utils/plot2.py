import os
from ast import literal_eval

import IPython
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st
import seaborn as sns
import torch
from cycler import cycler

import filesystem as fs
from misc import get_longest_sublists
from data_analysis import lookup_label

import plotting as plot

#experiment_ids = ['VarAdapt_MNIST', 'VarAdapt_CartPole']
#experiment_ids = ['MNIST_none_m0_lr025', 'MNIST_single_m0_lr025', 'MNIST_pl_m0_lr025', 'MNIST_pw_m0_lr025' ]
#experiment_ids = ['CartPole_none_m0_lr025', 'CartPole_single_m0_lr025', 'CartPole_per-layer_m0_lr025', 'CartPole_per-weight_m0_lr025']

#experiment_ids = ['MNIST_none_no_rt', 'MNIST_single_no_rt', 'MNIST_pl_no_rt', 'MNIST_pw_no_rt']
#experiment_ids = ['CartPole_none_no_rt', 'CartPole_single_no_rt', 'CartPole_per-layer_no_rt', 'CartPole_per-weight_no_rt']

#experiment_ids = ['MNIST_none_mx300', 'MNIST_single_mx300', 'MNIST_pl_mx300', 'MNIST_pw_mx300']
#experiment_ids = ['Seaquest_none_m0_lr025', 'Seaquest_single_m0_lr025', 'Seaquest_pl_m0_lr025', 'Seaquest_pw_m0_lr025']
this_file_dir_local1 = os.path.dirname(os.path.abspath(__file__))
package_root_this_file1 = fs.get_parent(this_file_dir_local1, 'es-rl')
d1 = os.path.join(package_root_this_file1, 'experiments', 'checkpoints')
experiment_ids = os.listdir(d1)

for experiment_id in experiment_ids:
    this_file_dir_local = os.path.dirname(os.path.abspath(__file__))
    package_root_this_file = fs.get_parent(this_file_dir_local, 'es-rl')
    d = os.path.join(package_root_this_file, 'experiments', 'checkpoints', experiment_id)
    directories = [os.path.join(d, di) for di in os.listdir(d) if os.path.isdir(os.path.join(d, di))]
    directories = [d for d in directories if 'monitoring' not in d and 'analysis' not in d]
    # Create result directory
    dst_dir = '/home/lorenzo/MEGA/UNI/MSc/Master\ Thesis/repo/graphics' + experiment_id + '-analysis'
    result_dir = os.path.join(d, experiment_id + '-analysis')
    
    for dirs in directories:
        plot.plot_stats(dirs + '/stats.csv', dirs)
    
        
