import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import sys

# Bart imports ############
if 'BART_TOOLBOX_PATH' in os.environ and os.path.exists(os.environ['BART_TOOLBOX_PATH']):
	sys.path.append(os.path.join(os.environ['BART_TOOLBOX_PATH'], 'python'))
elif 'TOOLBOX_PATH' in os.environ and os.path.exists(os.environ['TOOLBOX_PATH']):
	sys.path.append(os.path.join(os.environ['TOOLBOX_PATH'], 'python'))
else:
	raise RuntimeError("BART_TOOLBOX_PATH is not set correctly!")

import torch

def magnitude_only_sampling(kspace_in,factor_x,factor_y):
    len_x = kspace_in.shape[0]
    len_y = kspace_in.shape[1]
    full_kspace_mask = bart(1, 'poisson -Y {} -Z {} -y {} -z {} -C 38 -V 2 -e'.format(len_x,len_y,int(factor_x/1.9),int(factor_y/1.9))) 
    #38, 2, /1.9 for 1/2acc .... 22 8 and /1.1 for x4 acceleration
    undersampling_mask = np.squeeze(full_kspace_mask)
    tensor_mask = torch.tensor(undersampling_mask) 
    kspace_out = kspace_in * tensor_mask
    
    print( np.count_nonzero(undersampling_mask == 0))
    print( np.count_nonzero(undersampling_mask == 1))
    print(640*372)

    return  kspace_out,tensor_mask