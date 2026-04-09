
import os
import argparse
import h5py
import numpy as np
import torch
import sigpy as sp
import sigpy.mri as mr
import fastmri
from fastmri.data import transforms as Tf
from utils.transforms import coils_sampling, sense_combine_slice
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
import faulthandler
from M4RawDataset import recenter_via_demod
faulthandler.enable()

# fix for loading Namespace objects from saved checkpoints
class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

# Register Namespace for safe unpickling
import torch.serialization
torch.serialization.add_safe_globals({'Namespace': Namespace})

# New imports
from NexOP_model import NexOP
from ReconModule import ReconModule
from utils import complex_utils as cplx, transforms as T

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
eps = 1e-6


# example usage:
# python3 Compare_M4Raw.py --slice 10 --r 1.66 --dir_out ./assets --type T1 

def load_ckpt(path):
    try:
        ck = torch.load(path, map_location=device)
        return ck.get('model', ck)
    except AttributeError:
        # fallback to weights_only (safe load)
        ck = torch.load(path, map_location=device, weights_only=True)
        return ck.get('model', ck)

# paths (edit these to your actual files)
# 1st subject

DATA_FILES = [
    "./M4RawData/multicoil_csm_val/2022090301_T101.h5",
    "./M4RawData/multicoil_csm_val/2022090301_T102.h5",
    "./M4RawData/multicoil_csm_val/2022090301_T103.h5",
]

"""
# 2nd subject
DATA_FILES = [
    "./M4RawData/multicoil_csm_val/2022062303_T101.h5",
    "./M4RawData/multicoil_csm_val/2022062303_T102.h5",
    "./M4RawData/multicoil_csm_val/2022062303_T103.h5",
]
"""
"""
# T2w 1st subject
DATA_FILES = [
    "./M4RawData/multicoil_csm_val/2022090301_T201.h5",
    "./M4RawData/multicoil_csm_val/2022090301_T202.h5",
    "./M4RawData/multicoil_csm_val/2022090301_T203.h5",
]
"""
CHECKPOINTS = {
    "Poisson": {
        "recon": "./results/T1w/L2_checkpoints_poisson_Nex1_x{r}c/model_MoDL_Opt_30.pt" 
    },
    "Poisson3avg": {
        "recon": "./results/T1w/L2_checkpoints_poisson_Nex3_x{r}c/model_MoDL_Opt_30.pt"
    },
    "LOUPE1avg": {
        "mask": "./results/T1w/L2_checkpoints_LOUPE_Nex1_x{r}c/model_Mask_Opt_30.pt",    
        "recon": "./results/T1w/L2_checkpoints_LOUPE_Nex1_x{r}c/model_MoDL_Opt_30.pt",  
    },
    "LOUPE2avg": {
        "mask": "./results/T1w/L2_checkpoints_LOUPE_Nex2_x{r}c/model_Mask_Opt_30.pt",
        "recon": "./results/T1w/L2_checkpoints_LOUPE_Nex2_x{r}c/model_MoDL_Opt_30.pt",
    },
    "LOUPE3avg": {
        "mask": "./results/T1w/L2_checkpoints_LOUPE_Nex3_x{r}c/model_Mask_Opt_30.pt",
        "recon": "./results/T1w/L2_checkpoints_LOUPE_Nex3_x{r}c/model_MoDL_Opt_30.pt",
    },
    "NexOP": {
        "mask": "./results/T1w/L2_checkpoints_NexOP_x{r}c/model_Mask_Opt_30.pt",
        "recon": "./results/T1w/L2_checkpoints_NexOP_x{r}c/model_MoDL_Opt_30.pt",
    },
}


def load_and_preprocess(slice_idx):
    # load the three H5 files, extract k-spaces, combine via ESPIRiT -> single-coil images
    imgs_c = []
    csms = []
    for p in DATA_FILES:
        da = h5py.File(p, 'r')
        ks = np.array(da["kspace"])[slice_idx, :, :, ...]
        csm = np.array(da["csm"])[slice_idx, ...]
        ks_t = Tf.to_tensor(ks)
        img_c = fastmri.ifft2c(ks_t)
        csm_t = Tf.to_tensor(csm)
        
        imgs_c.append(img_c)  # append the k-space tensor directly
        csms.append(csm_t)  # append the csm tensor directly
    csms = torch.stack(csms, 0)  # [B,C,H,W,2]
    csm_t = csms[0]
    reps = torch.stack(imgs_c, 0)         # [3, H, W, 2]

    for r_i in range(0, reps.shape[0]):
        reps[r_i,...] = recenter_via_demod(reps[r_i,...].unsqueeze(0))  # [Nrep,C, H, W]\  

    target = reps
    # Norm CSM
    csm = torch.view_as_complex(csm_t)  # [B,1,C,H,W] -> [B,C,H,W]  
    norm_factor = (csm * csm.conj()).sum(dim = 0)   # [B,C,H,W]
    csm = csm / (norm_factor.unsqueeze(0)+eps)  # [B,C,H,W] -> [B,C,H,W]
    csm = torch.view_as_real(csm).unsqueeze(0)  # [B,C,H,W,2]
    return reps, target, csm

def run_unrolled(image_sampled, csm, mask_t, recon_ckpt, input_model):
    # load UnrolledModel with fallback to weights_only
    try:
        ck = torch.load(recon_ckpt, map_location=device)
    except AttributeError:
        ck = torch.load(recon_ckpt, map_location=device, weights_only=True)
    params = ck["params"]
    model_dict = ck["model"]
    model = ReconModule(n_layers=params.num_cnn_layers,k_iters=params.num_steps, input_model=input_model).to(device)
    model.load_state_dict(model_dict)
    model.eval()
    with torch.no_grad():
        inp = image_sampled.to(device)
        m   = mask_t.to(device)
        csm = csm.to(device)
        out = model(inp.float(), csm, mask=m)
    return out.permute(0,2,3,1).squeeze(0).cpu()

def make_poisson(kspace,csm, r, nex):
    if nex == 1:
        masks = []
        m = sp.mri.poisson(
            (256,195), r,
            calib=(20,20),
            dtype=float,
            crop_corner=False,
            return_density=False,
            seed=0,
            max_attempts=6,
            tol=0.1
        )
        acs =20
        m[256//2-acs//2-1:256//2+acs//2 + 1, 195//2-acs//2-1:195//2+acs//2+1] = 1.0 
        mpad = torch.tensor(m, dtype=torch.float32)
        m1 = torch.zeros_like(mpad)
        m2 = torch.zeros_like(mpad)
        masks.append(mpad)
        masks.append(m1)
        masks.append(m2)
        mask_stack = np.stack([mpad, m1, m2], axis=0)
        mask_torch = torch.tensor(mask_stack).float()

        masks = mask_torch.unsqueeze(0).unsqueeze(1).unsqueeze(-1).expand(1,4, 3, 256, 195,2).permute(0,2,1,3,4,5)
        #masks = torch.stack(masks, dim=0).unsqueeze(-0).unsqueeze(-1).expand(1,3, 256, 195,2) # [B,3,H,W]
        NexMap = mpad.clone()
        print(f"[DEBUG] masks shape: {masks.shape}, NexMap shape: {NexMap.shape}")
        print(f"[DEBUG] kspace shape: {kspace.shape}, csm shape: {csm.shape}")
        image_sampled, mask = coils_sampling(kspace[:,0,...], csm, masks, k_mag=True, mask_i= 0)  # [B,3,H,W,2]
        image_sampled = image_sampled.permute(0,3,1,2)  # [B,2,H,W]
    else:
        # generate nex independent masks at factor r*nex, then average
        masks = []
        for i in range(nex):
            if i == 0:
                acs = 20
            else:
                acs = 0
            m = sp.mri.poisson(
                (256,195), r*nex,
                calib=(acs,acs),
                dtype=float,
                crop_corner=False,
                return_density=False,
                seed=i,
                max_attempts=6,
                tol=0.1
            )
            if i == 0:
                m[256//2-acs//2-1:256//2+acs//2 + 1, 195//2-acs//2-1:195//2+acs//2+1] = 1.0  # center ACS
            masks.append(m)
        NexMap = torch.tensor(np.stack(masks, 0), dtype=torch.float32).sum(dim=0)
        M = torch.tensor(np.stack(masks, 0)).float() # [nex,H,W]
        masks = M.unsqueeze(0).unsqueeze(1).unsqueeze(-1).expand(1,4, 3, 256, 195,2).permute(0,2,1,3,4,5)
        print(f"[DEBUG] masks shape: {masks.shape}, NexMap shape: {NexMap.shape}")
        print(f"[DEBUG] kspace shape: {kspace.shape}, csm shape: {csm.shape}")
        image_sampled1, mask = coils_sampling(kspace[:,0,...], csm, masks, k_mag=False, mask_i= 0)
        image_sampled2, mask = coils_sampling(kspace[:,1,...], csm, masks, k_mag=False, mask_i= 1)
        image_sampled3, mask = coils_sampling(kspace[:,2,...], csm, masks, k_mag=False, mask_i= 2)
        image_sampled = torch.concat([image_sampled1, image_sampled2, image_sampled3], dim=-1)  # [B,3,H,W,2]
        image_sampled = image_sampled.permute(0,3,1,2)  # [B,2,H,W]
        return image_sampled, mask, NexMap
    return image_sampled, mask, NexMap

def make_loupe(kspace_base,csm, r, nex, ckpt_mask,r_str):
    # for nex>1, use the averaged kspace (first 2 for NEX2, first 3 for NEX3)
    r_effective = r
    if nex == 1:
        ks_in = kspace_base[:,0]
    elif nex == 2:
        ks_in = (kspace_base[:,0]+kspace_base[:,1])/2
        r_effective = r*2
    else:
        ks_in = (kspace_base[:,0]+kspace_base[:,1]+kspace_base[:,2])/3
        r_effective = r*3
    # load mask layer
    model_dict = load_ckpt(ckpt_mask.format(r=r_str))
    layer = NexOP(
                image_shape=(256, 195),R=r_effective,num_masks=1,sample_pattern='3D',num_acs_lines=20,init_method='uniform',device=device).to(device)
    
    layer.load_state_dict(model_dict)
    tau = 0.5
    mask = layer(tau).detach().cpu()       # [H,W]
    if nex == 1:
        m1 = torch.zeros_like(mask)
        m2 = torch.zeros_like(mask)
        masks = torch.concat([mask, m1, m2], dim=0).unsqueeze(-0).unsqueeze(-1).expand(1,3, 256, 195,2)  # [B,3,H,W,2]
        NexMap = mask[0,...] * nex
        image_sampled, mask = coils_sampling(kspace_base[:,0,...], csm, masks, k_mag=False, mask_i= 0)
        image_sampled = image_sampled.permute(0,3,1,2)  # [B,2,H,W]
    if nex == 2:
        m1 = torch.zeros_like(mask)
        masks = torch.concat([mask, mask, m1], dim=0).unsqueeze(-0).unsqueeze(-1).expand(1,3, 256, 195,2)  # [B,3,H,W,2]
        NexMap = mask[0,...] * nex
        image_sampled1, mask = coils_sampling(kspace_base[:,0,...], csm, masks, k_mag=False, mask_i= 0)
        image_sampled2, mask = coils_sampling(kspace_base[:,1,...], csm, masks, k_mag=False, mask_i= 1)
        image_sampled = torch.concat([image_sampled1, image_sampled2], dim=-1)  # [B,3,H,W,2]
        image_sampled = image_sampled.permute(0,3,1,2)  # [B,2,H,W]
    if nex == 3:
        masks = torch.concat([mask, mask, mask], dim=0).unsqueeze(-0).unsqueeze(-1).expand(1,3, 256, 195,2)  # [B,3,H,W,2]
        NexMap = mask[0,...] * nex
        image_sampled1, mask = coils_sampling(kspace_base[:,0,...], csm, masks, k_mag=False, mask_i= 0)
        image_sampled2, mask = coils_sampling(kspace_base[:,1,...], csm, masks, k_mag=False, mask_i= 1)
        image_sampled3, mask = coils_sampling(kspace_base[:,2,...], csm, masks, k_mag=False, mask_i= 2)
        image_sampled = torch.concat([image_sampled1, image_sampled2, image_sampled3], dim=-1)  # [B,3,H,W,2]
        image_sampled = image_sampled.permute(0,3,1,2)  # [B,2,H,W]
    prob_map = layer.get_prob_masks().detach().cpu()
    return image_sampled, mask, NexMap, prob_map

def make_jnop(reps, csm, r, ckpt_mask,r_str):
    # NexOP always uses all 3 reps
    model_dict = load_ckpt(ckpt_mask.format(r=r_str))
    layer = NexOP((256,195), R=r, num_masks=3, sample_pattern='3D',
                 num_acs_lines=20, init_method='random', device=device).to(device)
    layer.load_state_dict(model_dict)
    tau = 0.5

    # --- ADDED CODE START ---
    raw_masks = layer(tau).detach().cpu() # Shape: [3, 256, 195]
    matrix_size = 256 * 195 *3 
    
    print("\n" + "="*40)
    print(f"[NexOP] Acceleration Factors (Target R={r})")
    print("="*40)
    for i in range(raw_masks.shape[0]):
        mask_sum = raw_masks[i].sum().item()
        # Prevent division by zero just in case
        sampling_portion = mask_sum / matrix_size 
        print(f"Repetition {i+1}: Mask Sum = {mask_sum:.1f} | Sampling Portion = {sampling_portion:.3f} | Total samples = {matrix_size} | Effective R = {1/sampling_portion:.2f}")
    print("="*40 + "\n")
    
    # Now expand M as you were doing originally
    # M = raw_masks.unsqueeze(0).unsqueeze(-1).expand(1, 3, 256, 195, 2)
    # --- ADDED CODE END ---

    M = layer(tau).detach().cpu().unsqueeze(-0).unsqueeze(-1).expand(1,3, 256, 195,2) # [3,H,W]
    print(f"[DEBUG] M shape: {M.shape}")
    Mbig = M[0,:,...,-1]
    count = Mbig.sum(dim=0)

    image_sampled1, mask = coils_sampling(reps[:,0,...], csm, M, k_mag=True, mask_i= 0) # [B,H,W,2]
    image_sampled2, mask = coils_sampling(reps[:,1,...], csm, M, k_mag=True, mask_i= 1)# [B,H,W,2]
    image_sampled3, mask = coils_sampling(reps[:,2,...], csm, M, k_mag=True, mask_i= 2) # [B,H,W,2]
    image_sampled = torch.concat([image_sampled1, image_sampled2, image_sampled3], dim=-1)
    image_sampled = image_sampled.permute(0,3,1,2)  # [B,6,H,W]
    prob_map = layer.get_prob_masks().detach().cpu()


    # Add this to see the "continuous" acceleration factor
    continuous_sums = prob_map.sum(dim=[-1, -2]) 
    for i, c_sum in enumerate(continuous_sums):
        expected_accel = (256 * 195) / c_sum.item()
        sampling_portion = c_sum.item() / matrix_size
        print(f"Rep {i+1}- Prob Map Sum: {c_sum.item():.1f} | Sampling Portion: {sampling_portion:.3f}")

    return image_sampled, mask, count, prob_map

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--slice", type=int, required=True)
    p.add_argument("--r", type=float, required=True, help="acceleration factor")
    p.add_argument("--dir_out", type=str, required=False,default='./assets', help="save dir")
    p.add_argument("--type", type=str, required=False,default='T1', help="Contrast type, e.g. T1, T2")
    p.add_argument("--zoom_x",    type=int, default=150,
                   help="x coordinate of top-left corner of zoom square")
    p.add_argument("--zoom_y",    type=int, default=100,
                   help="y coordinate of top-left corner of zoom square")
    p.add_argument("--zoom_size", type=int, default=100,
                   help="size (width and height) of the zoom square")
    p.add_argument("--reps", type=int, choices=[0,1], default=1,
                   help="plot individual repetitions and target (1) or skip (0)")
    args = p.parse_args()
    if abs(args.r - 1.66) < 1e-3:   # handle float 1.66 safely
        r_str = "166"
    else:
        # for normal integer r values like 2 or 3, keep it as integer string
        r_str = str(int(args.r)) if args.r == int(args.r) else str(args.r)
    save_dir = args.dir_out
    reps, target, csm = load_and_preprocess(args.slice)
    print(f'Target size: {target.shape}')
    single_coil_input_rep1 = sense_combine_slice(target[0,...].unsqueeze(0), csm)
    single_coil_input_rep2 = sense_combine_slice(target[1,...].unsqueeze(0), csm)
    single_coil_input_rep3 = sense_combine_slice(target[2,...].unsqueeze(0), csm)
    target_single_coil_rep1 = fastmri.complex_abs(single_coil_input_rep1)
    target_single_coil_rep2 = fastmri.complex_abs(single_coil_input_rep2)
    target_single_coil_rep3 = fastmri.complex_abs(single_coil_input_rep3)
    target_single_coil = torch.stack([
        target_single_coil_rep1,
        target_single_coil_rep2,
        target_single_coil_rep3
    ], dim=1) # [B, 3, H, W]
    print(f"[DEBUG] target_single_coil shape: {target_single_coil.shape}")
    target_mag = target_single_coil.mean(dim=1).squeeze(0).cpu().numpy()  # [B, H, W]
    """
    if args.reps == 1:
        # plot individual repetitions and target image
        import os
        rep_imgs = []
        for i in range(reps.shape[0]):
            kspace = reps[i]
            # convert to numpy, pad, and convert back to tensor
            arr = kspace.cpu().numpy()
            H, W, _ = arr.shape
            pad_left = (256 - W) // 2
            pad_right = 256 - W - pad_left
            arr_p = np.pad(arr, ((0, 0), (pad_left, pad_right), (0, 0)), mode='constant', constant_values=0)
            kspace_p = torch.from_numpy(arr_p).to(kspace.device)
            im_p = fastmri.ifft2c(kspace_p)
            im_np = fastmri.complex_abs(im_p).cpu().numpy()
            rep_imgs.append(im_np)

    

        
        #### Comment it!! 
        kspace_t = fastmri.fft2c(target)
        arr_p = np.pad(kspace_t, ((0, 0), (pad_left, pad_right), (0, 0)), mode='constant', constant_values=0)
        kspace_p = torch.from_numpy(arr_p).to(kspace.device)
        im_p_t = fastmri.ifft2c(kspace_p)
        target_mag = fastmri.complex_abs(im_p_t).cpu().numpy()
        ####    

        rep_imgs.append(target_mag)
        reps_dir = os.path.join(save_dir, 'reps')
        os.makedirs(reps_dir, exist_ok=True)
        rep_concat = np.concatenate(rep_imgs, axis=1)
        fig_r, ax_r = plt.subplots(figsize=(4 * len(rep_imgs), 4))
        ax_r.imshow(rep_concat, cmap='gray')
        ax_r.axis('off')
        labels = [f'Rep{i+1}' for i in range(reps.shape[0])] + ['Target']
        for idx, lbl in enumerate(labels):
            x_pos = (idx + 0.5) / len(labels)
            ax_r.text(x_pos, 1.02, lbl, ha='center', va='bottom', transform=ax_r.transAxes)
        plt.tight_layout()
        fig_r.savefig(os.path.join(reps_dir, f'Reps_Slice{args.slice}_r{args.r}.png'), dpi=300)
        # plot log10 magnitude of k-space for repetitions and target
        logk_imgs = []
        # process each repetition k-space
        for i in range(reps.shape[0]):
            arr = reps[i].cpu().numpy()
            H, W, _ = arr.shape
            pad_left = (256 - W) // 2
            pad_right = 256 - W - pad_left
            arr_p = np.pad(arr, ((0, 0), (pad_left, pad_right), (0, 0)), mode='constant', constant_values=0)
            mag = np.sqrt(arr_p[..., 0]**2 + arr_p[..., 1]**2)
            logk = np.log10(mag + 1e-12)
            logk_imgs.append(logk)
        # target k-space
        k_target = fastmri.fft2c(target)
        arr_t = k_target.cpu().numpy()
        Ht, Wt, _ = arr_t.shape
        pad_left = (256 - Wt) // 2
        pad_right = 256 - Wt - pad_left
        arr_t_p = np.pad(arr_t, ((0, 0), (pad_left, pad_right), (0, 0)), mode='constant', constant_values=0)
        mag_t = np.sqrt(arr_t_p[..., 0]**2 + arr_t_p[..., 1]**2)
        logk_t = np.log10(mag_t + 1e-12)
        logk_imgs.append(logk_t)
        # concatenate and plot
        logk_concat = np.concatenate(logk_imgs, axis=1)
        fig_k, ax_k = plt.subplots(figsize=(4 * len(logk_imgs), 4))
        ax_k.imshow(logk_concat, cmap='gray')
        ax_k.axis('off')
        labels_k = [f'Rep{i+1}' for i in range(reps.shape[0])] + ['Target']
        for idx, lbl in enumerate(labels_k):
            x_pos = (idx + 0.5) / len(labels_k)
            ax_k.text(x_pos, 1.02, lbl, ha='center', va='bottom', transform=ax_k.transAxes)
        plt.tight_layout()
        fig_k.savefig(os.path.join(reps_dir, f'Reps_KspaceLog_Slice{args.slice}_r{args.r}.png'), dpi=300)
    # (you can insert your quantile‐based scaling here)
    """
    print(f"[DEBUG] After load_and_preprocess: reps={reps.shape}, target={tuple(target.shape)}, target={tuple(target.shape)}")
    # build a list of (name, ks, mask, recon_ckpt)
    jobs = []

    # Poisson 1 avg
    if args.r !=1:
        ks, m, NexMap = make_poisson(reps.unsqueeze(0), csm, args.r, nex=1)
        jobs.append(("Poisson", ks, m, CHECKPOINTS["Poisson"]["recon"].format(r=r_str), NexMap, csm, 'Poisson'))
    
    
    # Poisson 3 avg
    ks3, m3, NexMap = make_poisson(reps.unsqueeze(0), csm, args.r, nex=3)
    jobs.append(("Poisson3avg", ks3, m3, CHECKPOINTS["Poisson3avg"].get("recon").format(r=r_str),NexMap, csm, 'Poisson3avg'))
    
    # LOUPE 1,2,3 avg
    prob_map_LOUPE = [0 for _ in range(3)] 
    for nex in (1,2,3):
        if args.r != 1 or nex != 1:
            name = f"LOUPE{nex}avg"
            cfg  = CHECKPOINTS[name]
            ks_l, m_l, NexMap, prob_map_LOUPE[nex-1] = make_loupe(reps.unsqueeze(0), csm, args.r, nex, cfg["mask"],r_str)
            jobs.append((name, ks_l, m_l, cfg["recon"].format(r=r_str), NexMap, csm, name))

    
    # NexOP
    cfg = CHECKPOINTS["NexOP"]
    ks_j, m_j, NexMap, prob_map_NexOP = make_jnop(reps.unsqueeze(0), csm,  args.r, cfg["mask"],r_str)
    jobs.append(("NexOP", ks_j, m_j, cfg["recon"].format(r=r_str), NexMap, csm, 'NexOP'))
    
    job_names = [name for name, *_ in jobs]
    print(f"[DEBUG] Jobs to run: {job_names}")

    # reconstruct all
    recon_images = []
    for name, ks, mask, recon_ckpt,_, csm, input_model  in jobs:
        print(f"[DEBUG] Running reconstruction for: {name}")
        im_out = run_unrolled(ks, csm, mask, recon_ckpt, input_model)
        if (input_model == 'NexOP') or (input_model == 'LOUPE3avg') or (input_model == 'Poisson3avg'):
            final_recon = (fastmri.complex_abs(im_out[...,0:2]) + fastmri.complex_abs(im_out[...,2:4]) + fastmri.complex_abs(im_out[...,4:6]) ) / 3 
        elif (input_model == 'LOUPE2avg'):
            final_recon = (fastmri.complex_abs(im_out[...,0:2]) + fastmri.complex_abs(im_out[...,2:4])) / 2 
        else:
            final_recon = fastmri.complex_abs(im_out) 

        recon_images.append((name, final_recon))


    imgs = []
    # process reconstructed images
    for name, im in recon_images:
        print(f"[DEBUG] Processing image for {name}: shape={im.shape}")
        im_np = im.cpu().numpy()
        imgs.append(im_np)

    # process target image
    im_t_np = target_mag
    imgs.append(im_t_np)

    # concatenate horizontally
    print(f'im_t_np output shape {im_t_np.shape}')
    concat = np.concatenate(imgs, axis=1)

# display with labels and save
names = [name for name, _ in recon_images] + ["Target"]
fig, ax = plt.subplots(figsize=(4 * len(imgs), 4))
ax.imshow(concat, cmap='gray')
ax.axis('off')
# add method names above each segment
for idx, label in enumerate(names):
    # compute normalized x-position for the center of each 256-pixel segment
    x = (idx + 0.5) / len(names)
    ax.text(x, 1.02, label, ha='center', va='bottom', transform=ax.transAxes)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'Comparison_Slice' + str(args.slice) +'_r' +str(r_str)), dpi=300)

# Zoomed concat of square regions from each individual result + target
x0 = args.zoom_x
y0 = args.zoom_y
sz = args.zoom_size
x1 = x0 + sz
y1 = y0 + sz
# crop each image in imgs
zoomed_imgs = [img[y0:y1, x0:x1] for img in imgs]
# concatenate horizontally
zoom_concat = np.concatenate(zoomed_imgs, axis=1)
pad_max = np.ones((zoom_concat.shape[0],4)) * np.max(imgs[-1])
pad_min = np.ones((zoom_concat.shape[0],1)) * np.min(imgs[-1])
zoom_concat = np.concatenate([pad_min, pad_max, zoom_concat], axis=1)


fig_z, ax_z = plt.subplots(figsize=(4 * len(zoomed_imgs), 4))
ax_z.imshow(zoom_concat, cmap='gray')
ax_z.axis('off')
# add labels for each segment
names = [name for name, _ in recon_images] + ["Target"]
for idx, label in enumerate(names):
    x = (idx + 0.5) / len(names)
    ax_z.text(x, 1.02, label, ha='center', va='bottom', transform=ax_z.transAxes)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, f'Zoom_Slice{args.slice}_r{r_str}.png'), dpi=300)

print("Recon max/min:", imgs[0].max().item(), imgs[0].min().item())
print("Recon max/min:", imgs[1].max().item(), imgs[1].min().item())
#print("Recon max/min:", imgs[2].max().item(), imgs[2].min().item())
print("Target max/min:", target_mag.max().item(), target_mag.min().item())
print("Mask shape:", mask.shape)


# ==========================================
# plot folded-half masks (sum conjugate-symmetric halves)
# ==========================================
folded_imgs = []
plot_labels = [] # To hold the multi-line text for the plot

print("\n" + "="*65)
print(f"{'Method':<15} | {'Total Samples':<15} | {'Effective R':<15}")
print("="*65)

matrix_size = 256 * 195

for name, ks, mask, recon_ckpt, NexMap, csm, input_model in jobs:
    print(f"[DEBUG] Processing mask for {name}: shape={NexMap.shape}")
    
    # Calculate the exact total samples across all repetitions for this method
    total_samples = NexMap.sum().item()
    effective_r = matrix_size / total_samples if total_samples > 0 else 0.0
    
    # Print a clean row in the terminal
    print(f"{name:<15} | {total_samples:<15.1f} | {effective_r:<15.3f}")
    
    count_np = NexMap.cpu().numpy()
    folded = count_np
    Hm, Wm = folded.shape
    pad_left = (256 - Wm) // 2
    pad_right = 256 - Wm - pad_left
    folded_p = np.pad(folded, ((0, 0), (pad_left, pad_right)), mode='constant', constant_values=0)
    
    folded_imgs.append(folded_p)
    
    # Create a clean multi-line label to print directly above the plot segment
    plot_labels.append(f"{name}\nSpls: {int(total_samples)}\nEff R: {effective_r:.2f}")

print("="*65 + "\n")

# concatenate folded masks horizontally
fold_concat = np.concatenate(folded_imgs, axis=1)

# display and save folded mask montage
# Increased the figure height slightly (from 4 to 5) to fit the new multi-line text
fig_m, ax_m = plt.subplots(figsize=(4 * len(folded_imgs), 5)) 
im = ax_m.imshow(fold_concat, cmap='viridis', vmin=0, vmax=fold_concat.max(), interpolation='nearest')
ax_m.axis('off')

for idx, label in enumerate(plot_labels):
    x = (idx + 0.5) / len(plot_labels)
    # Added multi-line text support, keeping it centered
    ax_m.text(x, 1.02, label, ha='center', va='bottom', transform=ax_m.transAxes, fontsize=11)
    
fig_m.colorbar(im, ax=ax_m, label='Total samples (across all reps)', ticks=[0, 1, 2, 3])
plt.tight_layout()
plt.savefig(os.path.join(save_dir, f'Masks_Slice{args.slice}_r{r_str}.png'), dpi=300)

# Save the probability maps figure
print(f"[DEBUG] Plotting probability maps")
print(f"prob_map_LOUPE[0] shape: {prob_map_LOUPE[0].shape}")
print(f"prob_map_NexOP shape: {prob_map_NexOP.shape}")
prob_map_LOUPE1 = prob_map_LOUPE[0].squeeze(0)
prob_map_LOUPE2 = prob_map_LOUPE[1].squeeze(0)
prob_map_LOUPE3 = prob_map_LOUPE[2].squeeze(0)
prob_map_NexOP1 = prob_map_NexOP[0,:,:]
prob_map_NexOP2 = prob_map_NexOP[1,:,:]
prob_map_NexOP3 = prob_map_NexOP[2,:,:]

titles = ['LOUPE 1 avg', 'LOUPE 2 avg', 'LOUPE 3 avg',
          'NexOP Mask 1', 'NexOP Mask 2', 'NexOP Mask 3']

# Probability maps to plot (Assuming these Tensors/Arrays are defined and 2D: [H, W])
prob_maps = [prob_map_LOUPE1, prob_map_LOUPE2, prob_map_LOUPE3,
             prob_map_NexOP1, prob_map_NexOP2, prob_map_NexOP3]

# Convert all items in prob_maps to NumPy arrays, handling tensors
prob_maps_np = [(p.cpu().numpy() if torch.is_tensor(p) else p) for p in prob_maps]

# --- 1. Concatenation and Dimensioning ---

try:
    # Concatenate horizontally (side-by-side) along axis 1
    all_values = np.concatenate(prob_maps_np, axis=1)
except ValueError as e:
    print(f"Error: All maps must have the same height for concatenation. {e}")
    raise e

# Define dimensions for plotting
num_segments = len(prob_maps_np)
segment_width = prob_maps_np[0].shape[1] # Assumes all segments have the same width


# --- 2. Plotting the Concatenated Image (Fixes Critical Errors 1 & 2) ---

# Use num_segments (or num_segments * segment_width / 256 for normalized) for scaling
fig, ax = plt.subplots(figsize=(4 * num_segments, 5)) # Use num_segments for scaling

# CRITICAL FIX 1: Capture the image handle 'im' directly from ax.imshow()
im = ax.imshow(all_values, cmap='viridis', vmin=0, vmax=1, interpolation='nearest')

ax.axis('off')

# --- 3. Add Method Names (Labels) (Fixes Logic Error 3) ---

for idx, label in enumerate(titles):
    # Calculate the center x-coordinate in PIXEL units for the current segment
    x_center_pixel = (idx * segment_width) + (segment_width / 2)
    
    # Place text label above the plot using the total width for normalization 
    # and the calculated pixel position.
    ax.text(x_center_pixel, 
            -10, # Negative Y-coordinate places text slightly above the image data
            label, 
            ha='center', 
            va='bottom', 
            fontsize=12,
            weight='bold',
            transform=ax.transData) # Use transData for pixel coordinates

# --- 4. Add Single Colorbar ---

# CRITICAL FIX 2: Now 'im' is defined and can be passed to the colorbar function
cbar = fig.colorbar(im,
    ax=ax, # Attach the colorbar to the single axis
    orientation='vertical',
    shrink=0.75,
    aspect=30,
    label='Sampling Probability'
)

plt.tight_layout()
output_path = os.path.join(save_dir, f'Probability_Maps_Slice{args.slice}_r{r_str}.png')
plt.savefig(output_path, dpi=900)


# ==========================================
# Smooth Probability Maps 
# ==========================================
print("\n" + "="*65)
print("Generating Smoothed PDFs and Fitting to Asymmetric 2D Gaussian...")

# Define an asymmetric 2D Gaussian function for curve fitting
def gaussian_2d_asym(coords, a, x0, y0, sigma_x, sigma_y, offset):
    y, x = coords
    # Separate sigma_x and sigma_y allow for elliptical fits
    return a * np.exp(-(((x - x0)**2) / (2 * sigma_x**2) + ((y - y0)**2) / (2 * sigma_y**2))) + offset

k = 10 # Size of the k x k mean filter 
smooth_maps = []
fit_stds_x = []
fit_stds_y = []

for idx, pm in enumerate(prob_maps_np):
    smooth_pm = ndimage.uniform_filter(pm, size=k)
    smooth_maps.append(smooth_pm)
    
    # 1. Normalize the map so it sums to 1 (making it a true PDF)
    pdf = smooth_pm / np.sum(smooth_pm)
    
    h, w = pdf.shape
    y, x = np.mgrid[0:h, 0:w]
    
    # 2. Calculate Expected Values (Means)
    # E[X] and E[Y]
    E_x = np.sum(x * pdf)
    E_y = np.sum(y * pdf)
    
    # 3. Calculate Expected Value of Squares
    # E[X^2] and E[Y^2]
    E_x2 = np.sum((x**2) * pdf)
    E_y2 = np.sum((y**2) * pdf)
    
    # 4. Calculate Variance: Var(X) = E[X^2] - (E[X])^2
    var_x = E_x2 - (E_x**2)
    var_y = E_y2 - (E_y**2)
    
    # 5. True Standard Deviation
    std_x_val = np.sqrt(var_x)
    std_y_val = np.sqrt(var_y)
        
    fit_stds_x.append(std_x_val)
    fit_stds_y.append(std_y_val)
    print(f"{titles[idx]:<15} | True STD X: {std_x_val:.2f} | True STD Y: {std_y_val:.2f}")

# 3. Plotting the smooth maps with headlines
fig_smooth, axes = plt.subplots(1, len(smooth_maps), figsize=(4 * len(smooth_maps), 6), layout='constrained')
# Ensure axes is iterable if there's only one map
if len(smooth_maps) == 1:
    axes = [axes] 

for ax, smooth_pm, title, std_x, std_y in zip(axes, smooth_maps, titles, fit_stds_x, fit_stds_y):
    im_s = ax.imshow(smooth_pm, cmap='viridis', vmin=0, vmax=1, interpolation='nearest')
    ax.axis('off')
    
    # Create the headline with method name and both Gaussian stds
    if np.isnan(std_x) or np.isnan(std_y):
        headline = f"{title}\nFit Failed"
    else:
        headline = f"{title}\nSTD_x: {std_x:.1f} | STD_y: {std_y:.1f}"
        
    ax.set_title(headline, fontsize=12, fontweight='bold', pad=15)


# Add a single colorbar for the whole figure
cbar_s = fig_smooth.colorbar(
    im_s, 
    ax=axes, 
    orientation='vertical', 
    shrink=0.75, 
    aspect=30, 
    pad=0.02,
    fraction=0.05,
    label='Smooth Probability'
)


output_path_smooth = os.path.join(save_dir, f'Smooth_Probability_Maps_Slice{args.slice}_r{r_str}_type{args.type}.png')
plt.savefig(output_path_smooth, dpi=900)
print(f"Saved smoothed maps to: {output_path_smooth}")
print("="*65 + "\n")

