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

import utils.filesystem as fs
from utils.misc import get_longest_sublists
from utils.data_analysis import lookup_label

import utils.plotting as plot

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
                break

            with open(os.path.join(d3, 'init.log'), 'r') as f:
                s = f.read()
            #Check if naturgrad
            if 'per-weight' in s:
                plt.rc('font', family='sans-serif')
                plt.rc('xtick', labelsize='x-small')
                plt.rc('ytick', labelsize='x-small')
                
                figsize = mpl.figure.figaspect(9/16)
                
                # Computations/Transformations
                psuedo_start_time = stats['walltimes'].diff().mean()
                # Add pseudo start time to all times
                abs_walltimes = stats['walltimes'] + psuedo_start_time
                # Append pseudo start time to top of series and compute differences
                stats['time_per_generation'] = pd.concat([pd.Series(psuedo_start_time), abs_walltimes]).diff().dropna()
                stats['parallel_fraction'] = stats['workertimes']/stats['time_per_generation']
            
                # Compute moving averages
                for c in stats.columns:
                    if not 'Unnamed' in c and c[-3:] != '_ma':
                        stats[c + '_ma'] = stats[c].rolling(window=10, min_periods=1, center=True, win_type=None).mean()
                        
                nlines = 3
                # Loop over them and plot into same plot
                fig, ax = plt.subplots(figsize=figsize)
                ax.set_prop_cycle('color',plt.cm.tab20c(np.linspace(0,1,nlines)))
                
                print(d3)
                stats['sigma_avg'].plot(ax=ax, linestyle='None', marker='.', alpha=0.3, label='_nolegend_')
                stats['sigma_max'].plot(ax=ax, linestyle='None', marker='.', alpha=0.3, label='_nolegend_')
                stats['sigma_min'].plot(ax=ax, linestyle='None', marker='.', alpha=0.3, label='_nolegend_')
                
                stats['sigma_avg'].plot(ax=ax, linestyle='-', label='sigma_avg')
                stats['sigma_max'].plot(ax=ax, linestyle='--', alpha=0.6, label='sigma_max')
                stats['sigma_min'].plot(ax=ax, linestyle='--', alpha=0.6, label='sigma_min')
         
                box = ax.get_position()
                ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
                ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                
                plt.xlabel('Iteration')
                plt.ylabel('sigma')
                fig.savefig(os.path.join(d3, 'coolsigma.pdf'), bbox_inches='tight')
                plt.close(fig)
            else:
                pass
                
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
        

