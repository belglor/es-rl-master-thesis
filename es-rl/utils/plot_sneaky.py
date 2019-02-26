import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Plot settings
plt.rc('font', family='sans-serif')
plt.rc('xtick', labelsize='x-small')
plt.rc('ytick', labelsize='x-small')

figsize = (6.4, 4.8)

# Load data
try:
    stats = pd.read_csv('~/MEGA/UNI/MSc/Master Thesis/repo/es-rl-master-thesis/es-rl/experiments/checkpoints/Naturgrad_MNIST_outperformance/return_unp.csv')
except:
    pass

# Invert sign on negative returns (negative returns indicate a converted minimization problem)
#if (np.array(stats['return_max']) < 0).all():
#    for k in ['return_unp', 'return_avg', 'return_min', 'return_max', 'return_val']:
#        stats[k] = [-s for s in stats[k]]


# Compute moving averages
for c in stats.columns:
    if not 'Unnamed' in c and c[-3:] != '_ma':
        stats[c + '_ma'] = stats[c].rolling(window=5, min_periods=1, center=True, win_type=None).mean()
    
# Plot each of the columns including moving average
c_list = stats.columns.tolist()


# Find all c that are in this series
cis = {ci for ci in stats.columns if ci.split('_')[:-1] == c.split('_')[:-1] and not c[-3:] == '_ma'}
nlines = len(cis)
for ci in cis.difference({c}):
    c_list.remove(ci)
cis = sorted(list(cis))
c = ''.join(c.split('_')[:-1])
# Loop over them and plot into same plot
fig, ax = plt.subplots(figsize=figsize)
ax.set_prop_cycle('color',plt.cm.tab20(np.linspace(0,1,nlines)))
for ci in cis:
    stats[ci].plot(ax=ax, linestyle=None, marker='.', alpha=0, label='_nolegend_')
#ax.set_prop_cycle(None)
for ci in cis:
    print()
    stats[ci + '_ma'].plot(ax=ax, linestyle='-', label=ci, alpha=1, linewidth=1.0)
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
#plt.gca().set_xlim([270,300])
#plt.gca().set_ylim([0.85,1])

plt.xlabel('Iteration')

plt.ylabel('return_unp')
fig.savefig(os.path.join(chkpt_dir, c + '.pdf'), bbox_inches='tight')
plt.close(fig)
