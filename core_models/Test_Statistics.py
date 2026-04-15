
import os, sys
import logging
import numpy as np
import sigpy.plot as pl
import torch
import sigpy as sp
from scipy.ndimage import binary_closing
from scipy.ndimage import binary_fill_holes
import scipy.ndimage
import h5py
from torch.nn import functional as F
from torch.utils.data import DataLoader
import fastmri
from M4RawDataset import M4RawDataset
from NexMaskOpt.NexOP.core_models.ReconModule import ReconModule
# import custom libraries
from utils import transforms as T
from utils import complex_utils as cplx
# import custom classes
from subsample_fastmri import MaskFunc
from NexOP_model import NexOP
from torchmetrics.image.fid import FrechetInceptionDistance
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import piq
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
#### DEBUG ####
#import matplotlib.pyplot as plt


model = 'NexOP' # 'Poisson', 'LOUPE', 'NexOP'
contrast = 'T1w' # 'T1w', 'T2w' # Trained contrast
prefix_contrast = 'T1' # 'T1', 'T2' # Tested contrast
loss = 'l2' # 'l1', 'l2', 'ssim'
r = 1.66 # acceleration factor 
r_total = r
nex_number = 1
version = 'c' # '', 'b', 'c', 'd' or 'e'
SEED = 0 
acs_lines = 20
if loss == 'l1':
    loss_fn = 'L1'
elif loss == 'l2':
    loss_fn = 'L2'
elif loss == 'ssim':
    loss_fn = 'ssim'
# Set seed for reproducibility
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    
class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

# # Helper functions

def coils_sampling(inp: torch.Tensor, csm_c: torch.Tensor, masks: torch.Tensor, k_mag: bool = False, mask_i= 0):
    """
    kspace_c : torch.float32 tensor [C, H, W, 2]  (fastMRI style complex-last)
    csm_c    : torch.float32 tensor [C, H, W, 2]  (from cplx.to_tensor)
    returns:  torch.float32 tensor [H, W, 2] (complex-last)
    """
    # IFFT each coil
    img_c = inp
    # convert to complex dtype
    img_cplx = torch.view_as_complex(img_c.contiguous())   # [C,H,W]
    csm_cplx = torch.view_as_complex(csm_c.contiguous())   # [C,H,W]
    kspace_coil = fastmri.fft2c(torch.view_as_real(img_cplx)) # [H,W,2] (complex-last) 
    kspace_sampled = kspace_coil * masks[:,mask_i,...]
    image_sampled =  fastmri.ifft2c(kspace_sampled)  # [B,C,H,W,2] 
    image_sampled_cplx = torch.view_as_complex(image_sampled.contiguous())  # [B,C,H,W]
    im_out = torch.sum(image_sampled_cplx * (csm_cplx.conj()) , axis=1)
    im_out_real = torch.view_as_real(im_out) # [B,H,W,2] (complex-last)
    combined_mask = masks[...,0].clone()
    # back to real-imag pair
    return im_out_real, combined_mask                       # [H,W,2]


def sense_combine_slice(inp: torch.Tensor, csm_c: torch.Tensor, eps: float = 1e-6, k_mag: bool = False):
    """
    kspace_c : torch.float32 tensor [C, H, W, 2]  (fastMRI style complex-last)
    csm_c    : torch.float32 tensor [C, H, W, 2]  (from cplx.to_tensor)
    returns:  torch.float32 tensor [H, W, 2] (complex-last)
    """
    # IFFT each coil
    if k_mag == True:
        img_c = fastmri.ifft2c(inp)
    else:
        img_c = inp  # [C,H,W,2]
    # convert to complex dtype
    img_cplx = torch.view_as_complex(img_c.contiguous())   # [C,H,W]
    csm_cplx = torch.view_as_complex(csm_c.contiguous())   # [C,H,W]
    # numerator and denominator
    num   = (img_cplx * torch.conj(csm_cplx)).sum(dim=1)   # [H,W]
    out = num                              
    out = torch.view_as_real(out)  # [H,W,2] (complex-last)
    # back to real-imag pair
    return out                         # [H,W,2]


class DataTransform:
    """
    Data Transformer for training unrolled reconstruction models.
    """

    def __init__(self, mask_func, args, use_seed=False):
        self.mask_func = mask_func
        self.use_seed = use_seed
        self.rng = np.random.RandomState()

    def __call__(self, reps, target, csm):
        kspace_torch = fastmri.fft2c(reps)
        target_torch = target
  
        if model == 'Poisson':
            if nex_number == 1:
                mask2 = sp.mri.poisson((256,195), r, calib=(20,20), dtype=float, crop_corner=False, return_density=False, seed=0, max_attempts=6, tol=0.1)
                mask2b = np.zeros_like(mask2)
                mask2c = np.zeros_like(mask2)
                acs = 20
                mask2[256//2-acs//2:256//2+acs//2, 195//2-acs//2:195//2+acs//2] = 1.0 
                mask_stack = np.stack([mask2, mask2b, mask2c], axis=0)
                mask_torch = torch.tensor(mask_stack).float()
            elif nex_number == 3:
                mask2 = sp.mri.poisson((256,195), r*nex_number, calib=(20, 20), dtype=float, crop_corner=False, return_density=False, seed=0, max_attempts=6, tol=0.1)
                mask2b = sp.mri.poisson((256,195), r*nex_number, calib=(0, 0), dtype=float, crop_corner=False, return_density=False, seed=0, max_attempts=6, tol=0.1)
                mask2c = sp.mri.poisson((256,195), r*nex_number, calib=(0, 0), dtype=float, crop_corner=False, return_density=False, seed=0, max_attempts=6, tol=0.1)
                acs = 20
                mask2[256//2-acs//2:256//2+acs//2, 195//2-acs//2:195//2+acs//2] = 1.0 
                mask_stack = np.stack([mask2, mask2b, mask2c], axis=0)
                mask_torch = torch.tensor(mask_stack).float()
        else:
            mask2 = sp.mri.poisson((256,195), 2, calib=(20, 20), dtype=float, crop_corner=False, return_density=False, seed=0, max_attempts=6, tol=0.1)
            mask_padded = F.pad(torch.tensor(mask2),  (0, 0, 0, 0),  mode='constant', value=1)
            mask_torch = torch.stack([mask_padded.float(),mask_padded.float()],dim=2)

        return kspace_torch,target_torch,mask_torch,csm

def create_datasets(args):
    # Generate k-t undersampling masks
    train_mask = MaskFunc([0.08],[4])
    train_data = M4RawDataset(str(args.data_path), prefix=prefix_contrast, min_reps=3, transform = DataTransform(train_mask, args) )
    return train_data
def create_data_loaders(args):
    train_data = create_datasets(args)
#     print(train_data[0])

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

# %%
#Hyper parameters
params = Namespace()
params.data_path = "./M4RawData/multicoil_csm_test"
params.batch_size = 1
params.num_cnn_layers = 5
params.num_steps = 5
params.share_weights = True
params.modl_lamda = 0.05
params.lr = 0.0001 #*100
params.weight_decay = 0
params.lr_step_size = 20 #9
params.lr_gamma = 0.1
params.epoch = 31

train_loader = create_data_loaders(params)

if model == 'LOUPE':
    if nex_number == 1:
        in_model = 'LOUPE'
    if nex_number == 2:
        in_model = 'LOUPE2'
    elif nex_number == 3:
        in_model = 'LOUPE3'
elif model == 'Poisson':
    if nex_number == 1:
        in_model = 'Poisson'
    if nex_number == 3:
        in_model = 'Poisson3'
    
else:
    in_model = model

recon_model = ReconModule(n_layers=params.num_cnn_layers,k_iters=params.num_steps, input_model=in_model).to(device)
if r==1.66:
    r_name = '166'
else:
    r_name = str(r_total)
if model == 'NexOP':
##### NexOP Code #####
    mask_layer = NexOP(
        image_shape=(256, 195),R=r,num_masks=3,sample_pattern='3D',num_acs_lines=acs_lines,init_method='uniform',device=device).to(device)

    checkpoint_mask_file = "./results/"+contrast+"/"+loss_fn+"_checkpoints_NexOP_x"+r_name+version+"/model_Mask_Opt_30.pt"
    checkpoint_file = "./results/"+contrast+"/"+loss_fn+"_checkpoints_NexOP_x"+r_name+version+"/model_MoDL_Opt_30.pt"

    checkpoint_mask = torch.load(checkpoint_mask_file,map_location=device)
    checkpoint = torch.load(checkpoint_file,map_location=device)
    mask_layer.load_state_dict(checkpoint_mask['model'])  
    recon_model.load_state_dict(checkpoint['model'])
    mask_layer.eval() # Set mask layer to evaluation mode
elif model == 'LOUPE':

    if model == 'LOUPE':
        mask_layer = NexOP(
            image_shape=(256, 195),R=r*nex_number,num_masks=1,sample_pattern='3D',num_acs_lines=20,init_method='uniform',device=device).to(device)
    checkpoint_mask_file = "./results/"+contrast+"/"+loss_fn+"_checkpoints_LOUPE_Nex" +str(nex_number)+ "_x"+r_name+version+"/model_Mask_Opt_30.pt"
    checkpoint_file = "./results/"+contrast+"/"+loss_fn+"_checkpoints_LOUPE_Nex" +str(nex_number)+ "_x"+r_name+version+"/model_MoDL_Opt_30.pt"
    checkpoint_mask = torch.load(checkpoint_mask_file,map_location=device)
    checkpoint = torch.load(checkpoint_file,map_location=device)
    mask_layer.load_state_dict(checkpoint_mask['model'])  
    recon_model.load_state_dict(checkpoint['model'])
    mask_layer.eval() # Set mask layer to evaluation mode
elif model == 'Poisson':
    checkpoint_file = "./results/"+contrast+"/"+loss_fn+"_checkpoints_poisson_Nex"+str(nex_number)+"_x"+r_name+version+"/model_MoDL_Opt_30.pt"
    checkpoint = torch.load(checkpoint_file,map_location=device)
    recon_model.load_state_dict(checkpoint['model'])
    
recon_model.eval()  # Set model to evaluation mode 

output_dir = './cmmd-pytorch/generated_images'
target_dir = './cmmd-pytorch/reference_images'
dim1 = 256
dim2 = 195
# Load test data
test_loader = create_data_loaders(params)
slice = 1
eps = 1e-6
tau = 0.5
pad_left  = 30             # first: 30
pad_right = 256 - 224 - 1 # last: 224

def expand_to_rgb(image):
    """Convert a 2D grayscale image to 3D RGB with zeros in the other two channels."""
    return np.stack([image, np.zeros_like(image), np.zeros_like(image)], axis=-1)


# Initialize lists to store metrics
mse_in_list, mse_out_list = [], []
psnr_in_list, psnr_out_list = [], []
ssim_in_list, ssim_out_list = [], []
fsim_out_list = []
recon_sum, inp_sum, tar_sum_in,  tar_sum_out= torch.zeros((1,3,dim1,dim2)), torch.zeros((1,3,dim1,dim2)), torch.zeros((1,3,dim1,dim2)), torch.zeros((1,3,dim1,dim2))


for i in range(1):
    with torch.no_grad():  # Disable gradient computation for evaluation
        for data in test_loader:

            reps,target,mask,csm = data
            reps = reps.to(device)
            target = target.to(device)
            mask = mask.to(device)
            csm = csm.to(device)

            ### Normalize the sens maps ###
            csm = torch.view_as_complex(csm)  # [B,1,C,H,W] -> [B,C,H,W]  
            norm_factor = (csm * csm.conj()).sum(dim = 1)   # [B,C,H,W]
            csm = csm / (norm_factor.unsqueeze(1)+eps)  # [B,C,H,W] -> [B,C,H,W]
            csm = torch.view_as_real(csm)  # [B,C,H,W,2]


            if model == 'NexOP':
                ##### Layer code #####
                masks = mask_layer(tau)        
                masks = masks.unsqueeze(0).unsqueeze(-1).expand(reps.shape[0],reps.shape[2], reps.shape[1], 256, 195,2) 
                # average the repeated acquisitions
                masks = masks.permute(0,2,1,3,4,5)
                #############################

                # Forward pass through the model
                image_sampled1, combined_mask = coils_sampling(target[:,0,...], csm, masks, k_mag=True, mask_i= 0)  # [B,H,W,2]
                image_sampled2, combined_mask = coils_sampling(target[:,1,...], csm, masks, k_mag=True, mask_i= 1)  # [B,H,W,2]
                image_sampled3, combined_mask = coils_sampling(target[:,2,...], csm, masks, k_mag=True, mask_i= 2)  # [B,H,W,2]
                image_sampled = torch.concat([image_sampled1, image_sampled2, image_sampled3], dim=-1)
                image_sampled = image_sampled.permute(0,3,1,2)  # [B,6,H,W]
                input = (fastmri.complex_abs(image_sampled[:,0:2].permute(0,2,3,1)) + fastmri.complex_abs(image_sampled[:,2:4].permute(0,2,3,1)) + fastmri.complex_abs(image_sampled[:,4:6].permute(0,2,3,1))) / 3.0 # [B,6,H,W] -> [B,H,W] 

            elif model == 'LOUPE' and nex_number ==1:
                masks = mask_layer(tau).unsqueeze(0).unsqueeze(-1).expand(reps.shape[0],reps.shape[2], 256, 195,2)
                masks_zeros = torch.zeros_like(masks)
                masks = torch.stack([masks, masks_zeros, masks_zeros], dim=2)  # [B,3,C,H,W,2]
                masks = masks.permute(0,2,1,3,4,5)
                image_sampled, combined_mask = coils_sampling(target[:,0,...], csm, masks, k_mag=False, mask_i = 0)  # [B,H,W,2]
                #print('image_sampled shape before permute', image_sampled.shape)  # [B,2,H,W]
                image_sampled = image_sampled.permute(0,3,1,2)  # [B,2,H,W]
                input = fastmri.complex_abs(image_sampled.permute(0,2,3,1))

            elif model == 'LOUPE' and nex_number ==2:
                masks = mask_layer(tau).unsqueeze(0).unsqueeze(-1).expand(reps.shape[0],reps.shape[2], 256, 195,2)
                masks_zeros = torch.zeros_like(masks)
                masks = torch.stack([masks, masks, masks_zeros], dim=2)  # [B,3,C,H,W,2]
                masks = masks.permute(0,2,1,3,4,5)
                image_sampled1, combined_mask = coils_sampling(target[:,0,...], csm, masks, k_mag=False, mask_i = 0)  # [B,H,W,2]
                image_sampled2, combined_mask = coils_sampling(target[:,1,...], csm, masks, k_mag=False, mask_i = 1)  # [B,H,W,2]
                image_sampled = torch.concat([image_sampled1, image_sampled2], dim=-1)
                #print('image_sampled shape before permute', image_sampled.shape)  # [B,2,H,W]
                image_sampled = image_sampled.permute(0,3,1,2)  # [B,2,H,W]
                input = (fastmri.complex_abs(image_sampled[:,0:2].permute(0,2,3,1)) + fastmri.complex_abs(image_sampled[:,2:4].permute(0,2,3,1))) / 2.0

            
            elif model == 'LOUPE' and nex_number ==3:
                masks = mask_layer(tau).unsqueeze(0).unsqueeze(-1).expand(reps.shape[0],reps.shape[2], 256, 195,2)
                masks = torch.stack([masks, masks, masks], dim=2)  # [B,3,C,H,W,2]
                masks = masks.permute(0,2,1,3,4,5)
                image_sampled1, combined_mask = coils_sampling(target[:,0,...], csm, masks, k_mag=False, mask_i = 0)  # [B,H,W,2]
                image_sampled2, combined_mask = coils_sampling(target[:,1,...], csm, masks, k_mag=False, mask_i = 1)  # [B,H,W,2]
                image_sampled3, combined_mask = coils_sampling(target[:,2,...], csm, masks, k_mag=False, mask_i = 2)  # [B,H,W,2]
                image_sampled = torch.concat([image_sampled1, image_sampled2, image_sampled3], dim=-1)
                #print('image_sampled shape before permute', image_sampled.shape)  # [B,2,H,W]
                image_sampled = image_sampled.permute(0,3,1,2)  # [B,2,H,W]
                input = (fastmri.complex_abs(image_sampled[:,0:2].permute(0,2,3,1)) + fastmri.complex_abs(image_sampled[:,2:4].permute(0,2,3,1)) + fastmri.complex_abs(image_sampled[:,4:6].permute(0,2,3,1))) / 3.0 # [B,6,H,W] -> [B,H,W] 

            elif model == 'Poisson':
                masks = mask.unsqueeze(1).unsqueeze(-1).expand(reps.shape[0],reps.shape[2], reps.shape[1], 256, 195,2).permute(0,2,1,3,4,5)
                if nex_number == 1:
                    image_sampled, combined_mask = coils_sampling(target[:,0,...], csm, masks, k_mag=False, mask_i = 0)  # [B,H,W,2]
                    #print('image_sampled shape before permute', image_sampled.shape)  # [B,2,H,W]
                    image_sampled = image_sampled.permute(0,3,1,2)  # [B,2,H,W]
                    input = fastmri.complex_abs(image_sampled.permute(0,2,3,1))
                elif nex_number == 3:
                    image_sampled1, combined_mask = coils_sampling(target[:,0,...], csm, masks, k_mag=False, mask_i = 0)
                    image_sampled2, combined_mask = coils_sampling(target[:,1,...], csm, masks, k_mag=False, mask_i = 1)
                    image_sampled3, combined_mask = coils_sampling(target[:,2,...], csm, masks, k_mag=False, mask_i = 2)
                    image_sampled = torch.concat([image_sampled1, image_sampled2, image_sampled3], dim=-1)
                    image_sampled = image_sampled.permute(0,3,1,2)  # [B,2,H,W]
                    input = (fastmri.complex_abs(image_sampled[:,0:2].permute(0,2,3,1)) + fastmri.complex_abs(image_sampled[:,2:4].permute(0,2,3,1)) + fastmri.complex_abs(image_sampled[:,4:6].permute(0,2,3,1))) / 3.0 # [B,6,H,W] -> [B,H,W] 
            
            im_out = recon_model(image_sampled.float(),csm,mask=combined_mask).permute(0,2,3,1) # [B,H,W,2]

            # Build final image
            if in_model == 'NexOP' or in_model == 'LOUPE3' or in_model == 'Poisson3':
                # Produce final image by averaging the reults of all Nexes
                final_recon = (fastmri.complex_abs(im_out[...,0:2]) + fastmri.complex_abs(im_out[...,2:4]) + fastmri.complex_abs(im_out[...,4:6]) ) / 3.0 # [B,H,W]
                #final_recon = fastmri.complex_abs(im_out)
            elif in_model == 'LOUPE2':
                final_recon = (fastmri.complex_abs(im_out[...,0:2]) + fastmri.complex_abs(im_out[...,2:4]) ) / 2.0
                #final_recon = fastmri.complex_abs(im_out)
            else: 
                final_recon = fastmri.complex_abs(im_out) 


            ## Target image 
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

            # Handle outputs and inputs
            im_out = final_recon.cpu().squeeze(0)
            
            target = target[0,:,:].cpu().numpy() # Take the first in the batch - only 1 batch allowed

            #print(cplx.to_numpy(im_out.cpu()).shape)
            cplx_image_target = target_single_coil.squeeze(0).cpu().numpy()
            cplx_image_in = input.squeeze(0).cpu().numpy()
            cplx_image_out = im_out.cpu().squeeze(0).numpy()
    

            target_numpy = target_single_coil.cpu().squeeze(0).numpy()
            input_numpy = cplx_image_in
            out_numpy = cplx_image_out
            max_val_in = np.max(np.abs(np.concatenate((target_numpy,input_numpy),axis=0)))
            max_val_out = np.max(np.abs(np.concatenate((target_numpy,out_numpy),axis=0)))
            min_val_in = np.min(np.abs(np.concatenate((cplx_image_target,cplx_image_in),axis=0)))
            min_val_out = np.min(np.abs(np.concatenate((cplx_image_target,cplx_image_out),axis=0)))
            target_numpy_norm_in = np.abs(target_numpy)/max_val_in.squeeze(0)
            target_numpy_norm_out = np.abs(target_numpy)/max_val_out.squeeze(0)
            input_numpy_norm = np.abs(input_numpy)/max_val_in.squeeze(0)
            out_numpy_norm = np.abs(out_numpy)/max_val_out.squeeze(0)

        
            # Find comparison area:
            area = target_numpy_norm_in > 0.30
            kernel = np.ones((10, 10)) / 25.0
            #area = np.convolve(area, kernel, mode='constant', cval=0.0)
            area = scipy.ndimage.convolve(area.astype(float), kernel, mode='constant', cval=0.0)
            area[area>0.009] = 1
            structuring_element = np.ones((4,4))
            area = binary_closing(area, structure=structuring_element)
            area = binary_fill_holes(area)

            target_numpy_norm_in = target_numpy_norm_in * area
            target_numpy_norm_out = target_numpy_norm_out * area
            input_numpy_norm = input_numpy_norm * area
            out_numpy_norm = out_numpy_norm * area
            
            # Save for CMMD calculation
            target_rgb_out = expand_to_rgb(target_numpy_norm_out)
            target_rgb_in = expand_to_rgb(target_numpy_norm_in)
            output_rgb = expand_to_rgb(out_numpy_norm)
            
            slice = slice + 1

            ## Calculate metrics
            # Calculate SSIM values
            data_range_in = max_val_in - min_val_in
            data_range_out = max_val_out - min_val_out

            ssim_in, _ = ssim(target_numpy_norm_in, input_numpy_norm, data_range=data_range_in, full=True)
            ssim_out, _ = ssim(target_numpy_norm_out, out_numpy_norm, data_range=data_range_out, full=True)

            # Calculate PSNR
            psnr_in = T.PSNR_numpy(target_numpy_norm_in, input_numpy_norm)
            psnr_out = T.PSNR_numpy(target_numpy_norm_out, out_numpy_norm)

            # Calculate FID
            zeros_vec = torch.zeros((1,1,dim1,dim2))
            tar_in = torch.cat((cplx.to_tensor(np.abs(target_numpy_norm_in)).permute(2,0,1).unsqueeze(0),zeros_vec),dim=1)
            tar_sum_in = torch.cat((tar_sum_in,tar_in),dim=0)
            tar_out = torch.cat((cplx.to_tensor(np.abs(target_numpy_norm_out)).permute(2,0,1).unsqueeze(0),zeros_vec),dim=1)
            tar_sum_out = torch.cat((tar_sum_out,tar_out),dim=0)
            recon = torch.cat((cplx.to_tensor(np.abs(out_numpy_norm)).permute(2,0,1).unsqueeze(0),zeros_vec),dim=1)
            recon_sum = torch.cat((recon_sum,recon),dim=0)
            inp = torch.cat((cplx.to_tensor(np.abs(input_numpy_norm)).permute(2,0,1).unsqueeze(0),zeros_vec),dim=1) 
            inp_sum = torch.cat((inp_sum,inp),dim=0)       

            # Calculate MSE
            mse_in = np.mean(np.abs(input_numpy_norm-target_numpy_norm_in)**2)
            mse_out = np.mean(np.abs(out_numpy_norm-target_numpy_norm_out)**2)

            # Convert to PyTorch tensors and move to GPU
            target_tensor_out = torch.tensor(target_rgb_out, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device) * 2 - 1  # Normalize to [-1, 1]
            out_tensor = torch.tensor(output_rgb, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device) * 2 - 1


            # FSIM calculation
            fsim_score = piq.fsim(out_tensor+1, target_tensor_out+1, data_range=data_range_out)
            fsim_out_list.append(fsim_score.item())
            # Append metrics to lists
            mse_in_list.append(mse_in)
            mse_out_list.append(mse_out)
            psnr_in_list.append(psnr_in)
            psnr_out_list.append(psnr_out)
            ssim_in_list.append(ssim_in)
            ssim_out_list.append(ssim_out)



# Print average metrics
print(f'Average MSE output: {np.mean(mse_out_list):.4f} ± {np.std(mse_out_list):.4f}')
print(f'Average PSNR input: {np.mean(psnr_in_list):.4f}')
print(f'Average PSNR output: {np.mean(psnr_out_list):.4f}± {np.std(psnr_out_list):.4f}')
print(f'Average SSIM input: {np.mean(ssim_in_list):.4f}')
print(f'Average SSIM output: {np.mean(ssim_out_list):.4f} ± {np.std(ssim_out_list):.4f}')
print(f'Average FSIM output: {np.mean(fsim_out_list):.4f} ± {np.std(fsim_out_list):.4f}')
print(f'Test slices: {len(test_loader)}')


