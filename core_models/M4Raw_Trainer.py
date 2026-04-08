# %%

import os, sys
import logging
import argparse
import numpy as np
import torch
import sigpy as sp
from torch.nn import functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
import fastmri
from M4RawDataset import M4RawDataset
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for SSH/servers
import matplotlib.pyplot as plt
# import custom libraries
from utils import transforms as T
from utils import subsample as ss
from utils import complex_utils as cplx
# import custom classes
from utils.datasets import SliceData
from subsample_fastmri import MaskFunc
import argparse
from utils.transforms import coils_sampling, sense_combine_slice
from NexOP_model import NexOP
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


##### Usage example #####
# Script for training all methods
# python3 ./M4Raw_Trainer.py -r 3 -m NexOP -n 3 -l L2 -t 'T2' -g 2 -o ./results/T2w/L2_checkpoints_NexOP_x3c
#########################

p = argparse.ArgumentParser(description="Combine multi-coil k-space into single coil via ESPIRiT")
p.add_argument('-r','--acc_factor',  required=False, type=float, default=2, help="Acceleration factor")
p.add_argument('-o','--output_folder', required=True, help="Where to save single-coil .h5 files")
p.add_argument('-g','--gpu_number', required=False, default=1, help="The GPU number")
p.add_argument('-t','--scan_type', required=False, default='T1', help="T1 or T2")
p.add_argument('-m','--model', required=False, default='NexOP', help="NexOP / LOUPE / Poisson")
p.add_argument('-n','--nex_number',  required=False, type=float, default=1, help="Maximal number of Nex - For LOUPE and Poisson")
p.add_argument('-l', '--loss', required=False, default='L2', help="Loss function: L1 or L2")
args = p.parse_args()
scan_type = args.scan_type
r = args.acc_factor 
model_type = args.model
nex_number = args.nex_number
os.makedirs(args.output_folder, exist_ok=True)
exp_dir = args.output_folder 
device = torch.device(('cuda:'+args.gpu_number )if torch.cuda.is_available() else 'cpu')

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class DataTransform:
    """
    Data Transformer for training unrolled reconstruction models.
    """

    def __init__(self, mask_func, args, use_seed=False):
        self.mask_func = mask_func
        self.use_seed = use_seed
        self.rng = np.random.RandomState()

    def __call__(self, reps, target,csm):

        kspace_torch = fastmri.fft2c(reps)
        target_torch = target

        if model_type == 'Poisson':
            if nex_number == 1:
                mask2 = sp.mri.poisson((256,195), r, calib=(18,18), dtype=float, crop_corner=False, return_density=False, seed=0, max_attempts=6, tol=0.1)
                mask2b = np.zeros_like(mask2)
                mask2c = np.zeros_like(mask2)
                acs = 20
                mask2[256//2-acs//2-1:256//2+acs//2+1, 195//2-acs//2-1:195//2+acs//2+1] = 1.0 
                mask_stack = np.stack([mask2, mask2b, mask2c], axis=0)
                mask_torch = torch.tensor(mask_stack).float()
            elif nex_number == 3:
                mask2 = sp.mri.poisson((256,195), r*nex_number, calib=(18, 18), dtype=float, crop_corner=False, return_density=False, seed=0, max_attempts=6, tol=0.1)
                mask2b = sp.mri.poisson((256,195), r*nex_number, calib=(0, 0), dtype=float, crop_corner=False, return_density=False, seed=0, max_attempts=6, tol=0.1)
                mask2c = sp.mri.poisson((256,195), r*nex_number, calib=(0, 0), dtype=float, crop_corner=False, return_density=False, seed=0, max_attempts=6, tol=0.1)
                acs = 20
                mask2[256//2-acs//2-1:256//2+acs//2+1, 195//2-acs//2-1:195//2+acs//2+1] = 1.0 
                mask_stack = np.stack([mask2, mask2b, mask2c], axis=0)
                mask_torch = torch.tensor(mask_stack).float()
        else:
            mask2 = sp.mri.poisson((256,195), 2, calib=(20, 20), dtype=float, crop_corner=False, return_density=False, seed=0, max_attempts=6, tol=0.1)
            mask_padded = F.pad(torch.tensor(mask2),  (0, 0, 0, 0),  mode='constant', value=1)
            mask_torch = torch.stack([mask_padded.float(),mask_padded.float()],dim=2)
        return kspace_torch,target_torch,mask_torch,csm


def create_datasets(args):
    train_mask = MaskFunc([0.08],[4])
    train_data = M4RawDataset(str(args.data_path), prefix=scan_type, min_reps=3, transform = DataTransform(train_mask, args) )
    return train_data
def create_data_loaders(args):
    train_data = create_datasets(args)


    train_loader = DataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )

    return train_loader
def build_optim(args, params):
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    return optimizer

#Hyper parameters
params = Namespace()
params.data_path = "./M4RawData/multicoil_csm_train"
params.batch_size = 16
params.num_cnn_layers = 5#4
params.num_steps = 5#8
params.share_weights = True
params.modl_lamda = 0.05
params.lr = 0.0001  #*100
params.weight_decay = 0
params.lr_step_size = 14 #9
params.lr_gamma = 0.5
params.epoch = 31

# %%
train_loader = create_data_loaders(params)

# %%

from core_models.ReconModule import ReconModule
if model_type == 'Poisson' and nex_number == 3:
    recon_model = ReconModule(n_layers=params.num_cnn_layers,k_iters=params.num_steps, input_model='Poisson3').to(device)
elif model_type == 'LOUPE' and nex_number == 3:
    recon_model = ReconModule(n_layers=params.num_cnn_layers,k_iters=params.num_steps, input_model='LOUPE3').to(device)
elif model_type == 'LOUPE' and nex_number == 2:
    recon_model = ReconModule(n_layers=params.num_cnn_layers,k_iters=params.num_steps, input_model='LOUPE2').to(device)
else:
    print("Using model type:", model_type)
    recon_model = ReconModule(n_layers=params.num_cnn_layers,k_iters=params.num_steps, input_model=model_type).to(device)

##### NexOP Code #####
if model_type == 'NexOP':
    mask_layer = NexOP(
        image_shape=(256, 195),
        R=r,
        num_masks=3,
        sample_pattern='3D',
        num_acs_lines=20,
        init_method='uniform',
        device=device,
    ).to(device)
if model_type == 'LOUPE':
    mask_layer = NexOP(
        image_shape=(256, 195),
        R=r*nex_number,
        num_masks=1,
        sample_pattern='3D',
        num_acs_lines=20,
        init_method='uniform',
        device=device
    ).to(device)


######################## T1w hyperparams #####################################
divider = 2
divider_modl = 1
if model_type == "LOUPE" and nex_number == 1:
    divider = 4
if model_type == "LOUPE" and nex_number == 1 and r == 2:
    divider = 4
    divider_modl = 2
if model_type == "Poisson" and nex_number == 1:
    divider = 8
if model_type == "Poisson" and nex_number == 1 and r==3:
    divider_modl = 2
if model_type == "NexOP" and r == 4:
    divider_modl = 0.5 
    divider = 1.5 

if model_type == "Poisson" and nex_number == 3 and r==2:
    divider_modl = 3

if r == 1:
    divider = 2
    divider_modl = 2
if r ==1 and model_type == "Poisson" and nex_number == 3 :
    divider = 2
    divider_modl = 1
if r ==1 and model_type == "LOUPE" and nex_number == 3 :
    divider = 2
    divider_modl = 1


if r == 0.66 and model_type == "LOUPE" and nex_number == 3 :
    divider = 2
    divider_modl = 1
if r == 0.66 and model_type == "NexOP":
    divider = 4.5
    divider_modl = 3

block = 1
max_val =1
max_norm_recon = 1.0
max_norm_mask  = 0.5  
eta = 0

if args.loss == 'L2' and scan_type == 'T1':
    
    if r == 1 and model_type == "NexOP":
        divider = 3
        divider_modl = 1
    if r == 2 and model_type == "NexOP":

        divider = 3
        divider_modl = 1
    if r == 3 and model_type == "NexOP":
        divider = 2
        divider_modl = 0.5        
    if r == 1.66 and model_type == "NexOP":
        divider = 3
        divider_modl = 1


    if r == 1 and model_type == "LOUPE" and nex_number == 2:
        divider = 2
        divider_modl = 1
    if r == 1.66 and model_type == "LOUPE" and nex_number == 2:
        divider = 3
        divider_modl = 1.5
    if r == 2 and model_type == "LOUPE" and nex_number == 2:
        divider = 3
        divider_modl = 1.5
        
    if r == 3 and model_type == "LOUPE" and nex_number == 2:
        divider = 3
        divider_modl = 1

        divider = 3
        divider_modl = 1.5

    if r == 1.66 and model_type == "LOUPE" and nex_number == 3:
        divider = 2
        divider_modl = 1
        max_val = 0.1
    if r == 1 and model_type == "LOUPE" and nex_number == 3:
        divider = 3
        divider_modl = 1
    if r == 2 and model_type == "LOUPE" and nex_number == 3:
        divider = 2
        divider_modl = 1
        max_val = 0.1

    if r == 3 and model_type == "LOUPE" and nex_number == 3:
        divider = 3
        divider_modl = 1
        max_val = 0.1

    if r == 1.66 and model_type == "LOUPE" and nex_number == 1:
        divider = 3
        divider_modl = 0.5
        max_val = 0.1
    if r == 2 and model_type == "LOUPE" and nex_number == 1:
        divider = 3
        divider_modl = 0.5
        max_val = 0.1
    if r == 3 and model_type == "LOUPE" and nex_number == 1:
        divider = 3
        divider_modl = 1
        

    if r == 2 and model_type == "Poisson" and nex_number == 1:
        divider_modl = 2
    if r == 3 and model_type == "Poisson" and nex_number == 1:
        divider_modl = 2
    if r == 1.66 and model_type == "Poisson" and nex_number == 1:
        divider_modl = 2
        max_val = 1


    if r == 1.66 and model_type == "Poisson" and nex_number == 3:
        divider_modl = 1
        max_val = 1
    if r == 2 and model_type == "Poisson" and nex_number == 3:
        divider_modl = 1
        max_val = 1
    if r == 3 and model_type == "Poisson" and nex_number == 3:
        divider_modl = 1
        max_val = 1


######################## T2w hyperparams #####################################
if args.loss == 'L2' and scan_type == 'T2':
    
    if r == 1 and model_type == "NexOP":
        divider = 3
        divider_modl = 1
    if r == 2 and model_type == "NexOP":
        divider = 4
        divider_modl = 1
    if r == 3 and model_type == "NexOP":
        divider = 1
        divider_modl = 1
        max_val = 0.1
    if r == 1.66 and model_type == "NexOP":
        divider = 1
        divider_modl = 1


    if r == 1 and model_type == "LOUPE" and nex_number == 2:
        divider = 2
        divider_modl = 1
    if r == 1.66 and model_type == "LOUPE" and nex_number == 2:
        divider = 3
        divider_modl = 1.5
    if r == 2 and model_type == "LOUPE" and nex_number == 2:
        divider = 3
        divider_modl = 1.5
        
    if r == 3 and model_type == "LOUPE" and nex_number == 2:

        divider = 3
        divider_modl = 1.5

    if r == 1.66 and model_type == "LOUPE" and nex_number == 3:

        divider = 2
        divider_modl = 0.25

    if r == 1 and model_type == "LOUPE" and nex_number == 3:
        divider = 3
        divider_modl = 1
    if r == 2 and model_type == "LOUPE" and nex_number == 3:
        divider = 2
        divider_modl = 1

    if r == 3 and model_type == "LOUPE" and nex_number == 3:
        divider = 4
        divider_modl = 0.5
    if r == 1.66 and model_type == "LOUPE" and nex_number == 1:
        divider = 3
        divider_modl = 1
        
    if r == 2 and model_type == "LOUPE" and nex_number == 1:
        divider = 2
        divider_modl = 1
    if r == 3 and model_type == "LOUPE" and nex_number == 1:

        divider = 1
        divider_modl = 1
        
    if r == 2 and model_type == "Poisson" and nex_number == 1:
        divider_modl = 2
    if r == 3 and model_type == "Poisson" and nex_number == 1:
        divider_modl = 1
        eta = 1e-5
        block = 1
    if r == 1.66 and model_type == "Poisson" and nex_number == 1:
        divider_modl = 2


    if r == 1.66 and model_type == "Poisson" and nex_number == 3:
        divider_modl = 1
        max_val = 1
    if r == 2 and model_type == "Poisson" and nex_number == 3:
        divider_modl = 1
        max_val = 1
    if r == 3 and model_type == "Poisson" and nex_number == 3:
        divider_modl = 1
        max_val = 1



if model_type == 'NexOP' or model_type == 'LOUPE':
    optimizer = build_optim(params,list(recon_model.parameters()) + list(mask_layer.parameters())) 

    optimizer = torch.optim.Adam([
        {'params': recon_model.parameters(),     'lr': params.lr*10/divider_modl},
        {'params': mask_layer.parameters(),            'lr': params.lr*500*10/divider } 
    ])
else: 
    optimizer = build_optim(params,list(recon_model.parameters()))
    optimizer = torch.optim.Adam([
        {'params': recon_model.parameters(),     'lr': params.lr*10/divider_modl}
    ])

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, params.lr_step_size, params.lr_gamma)
if args.loss == 'L2':
    criterion = nn.MSELoss().to(device)  # MSE loss for complex images
elif args.loss == 'L1':
    criterion = nn.L1Loss().to(device)  # L1 loss for complex images

# ---- Debug plotting config ----
DEBUG_PLOTS = True          # set to False to disable all debug figures
DEBUG_EVERY = 25            # create a figure every N iterations
DEBUG_IDX = 0               # which element in the batch to visualize
DEBUG_SAVE_DIR = os.path.join(exp_dir, "debug_imgs")
os.makedirs(DEBUG_SAVE_DIR, exist_ok=True)



# %%
for epoch in range(params.epoch):
    recon_model.train()
    avg_loss = 0.
    tau = max(1.0 * (0.95 ** (epoch-1)), 0.1)
    eps = 1e-6
    for iter, data in enumerate(train_loader):
        reps, target, mask, csm = data
        reps = reps.to(device)
        target = target.to(device)
        mask = mask.to(device)
        csm = csm.to(device)
        ##### NexOP code #####
        pad_left  = 30             # first: 30
        pad_right = 256 - 224 - 1 # last: 224
        if model_type == 'NexOP':
            masks = mask_layer(tau)
            masks = masks.unsqueeze(0).unsqueeze(-1).expand(reps.shape[0],reps.shape[2], reps.shape[1], 256, 195,2)
            masks = masks.permute(0,2,1,3,4,5)

            
        elif model_type == 'LOUPE' and nex_number == 3:
            masks = mask_layer(tau).unsqueeze(0).unsqueeze(-1).expand(reps.shape[0],reps.shape[2], 256, 195,2)
            masks = torch.stack([masks, masks, masks], dim=2)  # [B,3,C,H,W,2]
            masks = masks.permute(0,2,1,3,4,5)

        elif model_type == 'LOUPE' and nex_number == 2:
            masks = mask_layer(tau).unsqueeze(0).unsqueeze(-1).expand(reps.shape[0],reps.shape[2], 256, 195,2)
            masks_zeros = torch.zeros_like(masks)
            masks = torch.stack([masks, masks, masks_zeros], dim=2)  # [B,3,C,H,W,2]
            masks = masks.permute(0,2,1,3,4,5)

        elif model_type == 'LOUPE':
            masks = mask_layer(tau).unsqueeze(0).unsqueeze(-1).expand(reps.shape[0],reps.shape[2], 256, 195,2)
            masks_zeros = torch.zeros_like(masks)
            masks = torch.stack([masks, masks_zeros, masks_zeros], dim=2)  # [B,3,C,H,W,2]
            masks = masks.permute(0,2,1,3,4,5)

        elif model_type == 'Poisson':
            masks = mask.unsqueeze(1).unsqueeze(-1).expand(reps.shape[0],reps.shape[2], reps.shape[1], 256, 195,2).permute(0,2,1,3,4,5)

        
        csm = torch.view_as_complex(csm)  # [B,1,C,H,W] -> [B,C,H,W]  
        #print("csm shape after view_as_complex", csm.shape)
        norm_factor = (csm * csm.conj()).sum(dim = 1)   # [B,C,H,W]
        #print("norm_factor shape", norm_factor.shape)
        denom = torch.clamp(norm_factor.real, min=eta).unsqueeze(1) + eps
        csm = csm / denom  # [B,C,H,W] -> [B,C,H,W]
        #print("csm shape after normalization", csm.shape)
        csm = torch.view_as_real(csm)  # [B,C,H,W,2]

        if model_type == 'NexOP' or (model_type == 'Poisson' and nex_number == 3) or (model_type == 'LOUPE' and nex_number == 3):
            image_sampled1, combined_mask = coils_sampling(target[:,0,...], csm, masks, k_mag=True, mask_i= 0)  # [B,H,W,2]
            image_sampled2, combined_mask = coils_sampling(target[:,1,...], csm, masks, k_mag=True, mask_i= 1)  # [B,H,W,2]
            image_sampled3, combined_mask = coils_sampling(target[:,2,...], csm, masks, k_mag=True, mask_i= 2)  # [B,H,W,2]
            image_sampled = torch.concat([image_sampled1, image_sampled2, image_sampled3], dim=-1)
            image_sampled = image_sampled.permute(0,3,1,2)  # [B,6,H,W]

        elif model_type == 'LOUPE' and nex_number == 2:
            image_sampled1, combined_mask = coils_sampling(target[:,0,...], csm, masks, k_mag=True, mask_i= 0)  # [B,H,W,2]
            image_sampled2, combined_mask = coils_sampling(target[:,1,...], csm, masks, k_mag=True, mask_i= 1)  # [B,H,W,2]
            image_sampled = torch.concat([image_sampled1, image_sampled2], dim=-1)
            image_sampled = image_sampled.permute(0,3,1,2)  # [B,6,H,W]
        else:
            image_sampled, combined_mask = coils_sampling(target[:,0,...], csm, masks, k_mag=False, mask_i = 0)  # [B,H,W,2]
            image_sampled = image_sampled.permute(0,3,1,2)  # [B,2,H,W]

                
                
        if model_type == 'Poisson':
            combined_mask = combined_mask.clone()            
            
        
        ### Forward model ###
        im_out = recon_model(image_sampled.float(),csm,mask=combined_mask).permute(0,2,3,1) # [B,H,W,2]

        # Build final image
        if model_type == 'NexOP' or (model_type == 'Poisson' and nex_number == 3) or (model_type == 'LOUPE' and nex_number == 3):
            # Produce final image by averaging the reults of all Nexes
            final_recon = (fastmri.complex_abs(im_out[...,0:2]) + fastmri.complex_abs(im_out[...,2:4]) + fastmri.complex_abs(im_out[...,4:6]) ) / 3 # [B,H,W]
            #final_recon = fastmri.complex_abs(im_out) 
        elif model_type == 'LOUPE' and nex_number == 2:
            final_recon = (fastmri.complex_abs(im_out[...,0:2]) + fastmri.complex_abs(im_out[...,2:4])) / 2 # [B,H,W]
            #final_recon = fastmri.complex_abs(im_out) 
        else: 
            final_recon = fastmri.complex_abs(im_out) 
        #print(kspace_avg.shape, combined_mask.shape, im_out.shape)
        #print('target shape', target.shape)
        
        ###### Target is single coil averaged over NEX image ######
        single_coil_input_rep1 = sense_combine_slice(target[:,0,...], csm)
        single_coil_input_rep2 = sense_combine_slice(target[:,1,...], csm)
        single_coil_input_rep3 = sense_combine_slice(target[:,2,...], csm)

        target_single_coil_rep1 = fastmri.complex_abs(single_coil_input_rep1)
        target_single_coil_rep2 = fastmri.complex_abs(single_coil_input_rep2)
        target_single_coil_rep3 = fastmri.complex_abs(single_coil_input_rep3)
        #print('target_single_coil_rep1 shape', target_single_coil_rep1.shape)
        target_single_coil = torch.stack([
            target_single_coil_rep1,
            target_single_coil_rep2,
            target_single_coil_rep3
        ], dim=1)  # [B, 3, H, W]
        target_single_coil = target_single_coil.mean(dim=1)  # [B, H, W]

        # ---- Debug plot ----
        if DEBUG_PLOTS and (iter % DEBUG_EVERY == 0) and (epoch == 0 or epoch == 7 or epoch == 20 or epoch == 29):
            with torch.no_grad():
                b = DEBUG_IDX if DEBUG_IDX < reps.size(0) else 0

                # Input magnitude (ReconModule input is [B,2,H,W]; convert to complex-last first)
                if model_type == 'NexOP' or (model_type == 'Poisson' and nex_number == 3) or (model_type == 'LOUPE' and nex_number == 3):
                    inp_avergae = (fastmri.complex_abs(image_sampled[b,0:2].permute(1,2,0)) + fastmri.complex_abs(image_sampled[b,2:4].permute(1,2,0)) + fastmri.complex_abs(image_sampled[b,4:6].permute(1,2,0))) / 3
                    inp_mag  = inp_avergae.detach()
                elif model_type == 'LOUPE' and nex_number == 2:
                    inp_avergae = (fastmri.complex_abs(image_sampled[b,0:2].permute(1,2,0)) + fastmri.complex_abs(image_sampled[b,2:4].permute(1,2,0)) ) / 2
                    inp_mag  = inp_avergae.detach()
                else:
                    inp_mag  = fastmri.complex_abs(image_sampled[b].clone().detach().permute(1,2,0))

                # Output & target are already complex-last [B,H,W,2]
                out_mag  = final_recon[b].clone().detach()
                tgt_mag  = target_single_coil[b].clone().detach()

                CSM = fastmri.complex_abs(csm.clone().detach())
                csm_complex = torch.view_as_complex(csm[b,3,:,:,:].clone().detach())
                csm_phase = torch.angle(csm_complex)
                in_phase = torch.angle(torch.view_as_complex(image_sampled[b,0:2].permute(1,2,0).clone().detach()))
                fig, axs = plt.subplots(1, 9, figsize=(12, 4))
                titles = ['Input to ReconModule', 'ReconModule Output', 'Target', 'Mask1', 'Mask2', 'Mask3]', 'CSM Magnitude', 'CSM Phase', 'In phase']
                for ax, img, title in zip(axs, [inp_mag, out_mag, tgt_mag, masks[0,0,0, ..., 0], masks[0,1,0, ..., 0], masks[0,2,0, ..., 0,], CSM[b,3,:,:], csm_phase, in_phase], titles):
                    ax.imshow(img.cpu().numpy(), cmap='gray')
                    cbar = fig.colorbar(ax.images[-1], ax=ax, fraction=0.046, pad=0.04)
                    ax.set_title(title)
                    ax.axis('off')
                fig.tight_layout()
                save_path = os.path.join(DEBUG_SAVE_DIR, f'ep{epoch:03d}_it{iter:05d}.png')
                fig.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=0)
                plt.close(fig)
                # Additional concatenated plot of input, output, and target
                inp_np = inp_mag.cpu().numpy()
                out_np = out_mag.cpu().numpy()
                tgt_np = tgt_mag.cpu().numpy()
                concat = np.concatenate([inp_np, out_np, tgt_np], axis=1)
                vmin, vmax = concat.min(), concat.max()
                fig_cat, ax_cat = plt.subplots(1, 1, figsize=(12, 4))
                ax_cat.imshow(concat, cmap='gray', vmin=vmin, vmax=vmax)
                ax_cat.set_title('Input | Output | Target')
                ax_cat.axis('off')
                save_path_cat = os.path.join(DEBUG_SAVE_DIR, f'concat_ep{epoch:03d}_it{iter:05d}.png')
                fig_cat.savefig(save_path_cat, dpi=150, bbox_inches='tight', pad_inches=0)
                plt.close(fig_cat)
        
        #print("im_out shape", im_out.shape)
        #print("rss shape", rss.shape)
        # --- Sanity check: make sure the model output still tracks gradients ---
        if not final_recon.requires_grad:
            print("\n⚠️ final_recon.requires_grad is False — output seems detached from the graph.")
            print("  • im_out.requires_grad:", getattr(im_out, 'requires_grad', None))
            # Check a couple of representative parameters
            some_params = []
            for name, p in recon_model.named_parameters():
                if p.requires_grad:
                    some_params.append(name)
                    if len(some_params) >= 5:
                        break
            print("  • Example trainable params:", some_params)
            print("  • HINT: Look inside ReconModule.forward for any `.detach()`, `.data`, `with torch.no_grad():`, or NumPy ops that would break autograd.")
        loss = criterion(final_recon,target_single_coil)
        optimizer.zero_grad()
        #k=l
        # Check for parameters without grad tracking
        no_grad_params = [name for name, p in recon_model.named_parameters() if not p.requires_grad]
        if model_type in ['NexOP', 'LOUPE']:
            no_grad_params += [f"mask_layer.{name}" for name, p in mask_layer.named_parameters() if not p.requires_grad]

        if no_grad_params:  # Only print if something has no grad
            print("⚠️ Parameters without grad:", no_grad_params)

        # Also check if loss itself has grad_fn
        if loss.grad_fn is None:
            print("⚠️ Loss has no grad_fn — may be detached from the graph!")

        loss.backward()

        if block == 1:
            ##### Nan handling #####
            # ---- Gradient clipping & NaN guard ----

            if model_type in ['NexOP', 'LOUPE']:
                torch.nn.utils.clip_grad_norm_(recon_model.parameters(), max_norm_recon)
                torch.nn.utils.clip_grad_norm_(mask_layer.parameters(),  max_norm_mask)
                for p in mask_layer.parameters():
                    if p.grad is not None:
                        p.grad.data.clamp_(-max_val, max_val)
            else:
                torch.nn.utils.clip_grad_norm_(recon_model.parameters(), max_norm_recon)

            # Detect NaNs/Infs before optimizer.step()
            bad = False
            for p in (list(recon_model.parameters()) +
                    (list(mask_layer.parameters()) if model_type in ['NexOP','LOUPE'] else [])):
                if p.grad is not None:
                    if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                        bad = True
                        break

            if bad or torch.isnan(loss).any() or torch.isinf(loss).any():
                logging.warning("NaN/Inf detected — skipping optimizer.step()")# and halving LR.")
                #for pg in optimizer.param_groups:
                #    pg['lr'] *= 0.5
                optimizer.zero_grad(set_to_none=True)
                continue  # skip this batch


        optimizer.step()
        #diff = (mask_layer.logits - before).abs().mean().item()
        #print("mean Δlogit:", diff)
        avg_loss = 0.99 * avg_loss + 0.01 * loss.item() if iter > 0 else loss.item()
        current_lr1 = scheduler.get_last_lr()[0]
        #current_lr2 = scheduler.get_last_lr()[1]
        #k = l
        if iter % 23 == 0:
            logging.info(
                f'Epoch = [{epoch:3d}/{params.epoch:3d}] '
                f'Iter = [{iter:4d}/{len(train_loader):4d}] '
                f'Loss = {loss.item():.4g} Avg Loss = {avg_loss:.4g} learning-rates = {current_lr1:.4g} '
            ) 
        

    
    scheduler.step()

    #Saving the model
    if epoch % 15 == 0:
        torch.save(
            {
                'epoch': epoch,
                'params': params,
                'model': recon_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'exp_dir': exp_dir
            },
            f=os.path.join(exp_dir, 'model_MoDL_Opt_%d.pt'%(epoch))
        )
        if model_type == 'NexOP' or model_type == 'LOUPE':
            torch.save(
                {
                    'epoch': epoch,
                    'params': params,
                    'model': mask_layer.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'exp_dir': exp_dir
                },
                f=os.path.join(exp_dir, 'model_Mask_Opt_%d.pt'%(epoch))
            )
        
        


