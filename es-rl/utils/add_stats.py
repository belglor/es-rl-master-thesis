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

this_file_dir_local = os.path.dirname(os.path.abspath(__file__))
package_root_this_file = fs.get_parent(this_file_dir_local, 'es-rl')
d1 = os.path.join(package_root_this_file, 'experiments', 'checkpoints')
experiments_ids = os.listdir(d1)

#Loop over experiment folders
for experiments in experiments_ids:
    d2 = os.path.join(d1,experiments)
    dirs = os.listdir(d2)
    #Loop over experiments settings
    for ds in dirs:
        #If not an experiment, skip
        if ds[-8:] == "analysis":
            pass
        else:
            d3 = os.path.join(d2,ds)
            csvfile_path = os.path.join(d3, 'stats.csv')
            try:
                stats = pd.read_csv(csvfile_path)
            except:
                print("The experiment "+str(d3)+ "has no stats file")
            rescale = 0.25
            #Check if column is already present
            try:
                check = stats['param_change']
                pass
            except:                
                #Read init file to check for options
                with open(os.path.join(d3, 'init.log'), 'r') as f:
                    s = f.read()
                #Check if naturgrad
                if 'Use natural gradient  True' in s:
                    rescale = 5
                #Create param_change column
                grad_norm = stats['grad_norm']
                param_change = grad_norm * rescale
                #Add column            
                stats['param_change'] = param_change 
                stats.to_csv(csvfile_path)
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
        

