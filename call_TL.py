import numpy as np
from pdb import set_trace as bp

import sys
sys.path.append('./Venus_Detectability/')
import compute_TL_modules as ctm

## Discretization
delta_dist = 50e3
epsilon = 5e3

delta_depth = 5e3
#delta_depth = 25e3
depths = np.arange(5e3, 50e3+delta_depth, delta_depth)

## STF
#period = 1./1e-1
#stf = gf.BoxcarSTF(period, anchor=0.)
#stf = gf.BoxcarSTF(period, anchor=0.)
#stf = gf.TriangularSTF(effective_duration=period)
stf = None

bins = np.logspace(np.log10(1e-2), np.log10(1), 4)
f_targets = []
for ibin, (binleft, binright) in enumerate(zip(bins[:-1], bins[1:])):
    if ibin == 0:
        binleft = None
    if ibin == len(bins)-2:
        binright = None
    f_targets += [[binleft, binright]]

## Greens functions STORES
base_folder = '/projects/restricted/infrasound/data/infrasound/2023_Venus_inversion/'

l_stores = ['Hot10', 'Hot25', 'Cold100']
l_stores = ['Cold100']
#l_comp = [True, False]
l_comp = [True]

dists = []
#dists.append(np.arange(0., 50.e3, 5e3)) # in km
dists.append(np.arange(50.e3+epsilon, 8000.e3+epsilon, delta_dist)) # in km
dists.append(np.arange(8000.e3+epsilon, 16000.e3+epsilon, delta_dist)) # in km
"""
dists.append(np.arange(0., 50.e3, 5e3)) # in km
dists.append(np.arange(50.e3+epsilon, 8000.e3+epsilon, delta_dist)) # in km
dists.append(np.arange(8000.e3+epsilon, 16000.e3+epsilon, delta_dist)) # in km
dists.append(np.arange(0., 50.e3, 5e3)) # in km
dists.append(np.arange(50.e3+epsilon, 8000.e3+epsilon, delta_dist)) # in km
dists.append(np.arange(8000.e3+epsilon, 16000.e3+epsilon, delta_dist)) # in km
"""

for displacement in l_comp:
    for store in l_stores:

        stores_id = []
        #stores_id.append(f'GF_venus_{store}_qssp_nearfield')
        stores_id.append(f'GF_venus_{store}_qssp')
        stores_id.append(f'GF_venus_{store}_qssp_8000km')

        all_amps_RW = ctm.get_all_amps(base_folder, stores_id, dists, depths, f_targets, stf, displacement)
        str_comp = '_u' if displacement else '_v'
        all_amps_RW.to_csv(f'{base_folder}GF_Dirac_1Hz_all_wfreq_{store}{str_comp}.csv', header=True, index=False)