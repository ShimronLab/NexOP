import os
# Disable HDF5 file locking to avoid BlockingIOError when writing
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
import glob
import argparse
import utils.complex_utils as cplx
import h5py

def compute_norm_factor(file_list):
    # Load raw k-space arrays, average them, and compute 95th percentile magnitude
    kspaces = [np.array(h5py.File(fn, 'r')['kspace']) for fn in file_list]
    avg_k = cplx.to_tensor(np.mean(np.stack(kspaces, axis=0), axis=0))

    S, C, H, W, _ = avg_k.shape
    w_crop = 24
    left = (W - w_crop) // 2
    right = left + w_crop
        
    kcrop = avg_k[:, :, :, left:right, :]        # [B, H, 24, 2]
    kcrop_ch = kcrop.permute(0,1,4,2,3)       # [B, 2, H, 24]
    pad = (left, W - right, 0, 0)           # pad W dimension only
    kpad_ch = F.pad(kcrop_ch, pad, mode='constant', value=0)  # [B,2,H,W]
    kpad = kpad_ch.permute(0,1,3,4,2)         # [B, H, W, 2]

    # Normalize the data with 95% percentile
    im_lowres = fastmri.rss_complex(fastmri.ifft2c(kpad).permute(1,0,2,3,4))
    mag_vals = im_lowres.reshape(-1)
    try:
        scale = torch.quantile(mag_vals, 0.95)
    except AttributeError:
        total_vals = mag_vals.numel()
        idx = total_vals - int(round(0.05 * total_vals))
        scale = mag_vals.kthvalue(idx).values
    # Fallback to max if quantile fails or is zero
        if scale == 0:
            scale = mag_vals.max()
    
    return scale

import numpy as np
import torch
import fastmri
from fastmri.data import transforms as T
import sigpy.mri as mr  
import sigpy as sp
import torch.nn.functional as F

def process_file(in_path: str, out_path: str, norm_factor: float):
    # --- load multi-coil k-space: assume shape [S, C, H, W] or [S, C, H, W, 2] ---
    with h5py.File(in_path, 'r') as f:
            # Compute the 95th percentile for normalization
        kspace_all = cplx.to_tensor(np.array(f['kspace']))  # keep original representation
        kspace_cut = kspace_all[...,30:225,:]
    sens_slices = []
    kspaces = []
    scale = norm_factor  # Use the precomputed normalization factor
    
    for sl in range(kspace_all.shape[0]):
        
        mag_vals = norm_factor[...].reshape(-1)
        try:
            scale = torch.quantile(norm_factor, 0.95)
        except AttributeError:
            total_vals = mag_vals.numel()
            idx = total_vals - int(round(0.05 * total_vals))
            scale = mag_vals.kthvalue(idx).values
        # Fallback to max if quantile fails or is zero
            if scale == 0:
                scale = mag_vals.max()
               
        # Compute the 95th percentile for normalization
        """
        try:
            scale = torch.quantile(magnitude_vals, 0.95)
        except AttributeError:
            total_vals = magnitude_vals.numel()
             idx = total_vals - int(round(0.05 * total_vals))
             scale = magnitude_vals.kthvalue(idx).values
        # Fallback to max if quantile fails or is zero
        if scale == 0:
             scale = magnitude_vals.max()
        """
        kslice = kspace_cut[sl]
        slice_image = fastmri.ifft2c(kslice)
        slice_image_abs = fastmri.complex_abs(slice_image)
        slice_image_rss = fastmri.rss(slice_image_abs)
        image_slice_norm = (slice_image ) / scale#/(x_max-x_min)
        kslice = fastmri.fft2c(image_slice_norm)  # re-transform to k-space
        # Convert torch tensor to numpy complex array for SigPy
        kslice_np = cplx.to_numpy(kslice)
        # estimate ESPIRiT maps
        sens = mr.app.EspiritCalib(kslice_np, max_iter=20, crop=0.95, kernel_width=7, show_pbar=False).run()  # [C, H, W]

        sens_slices.append(sens)
        kspaces.append(kslice_np)

    csm = np.stack(sens_slices, axis=0)  # [S, C, H, W]
    kspace_all = np.stack(kspaces, axis=0)  # [S, C, H, W] or [S, C, H, W, 2]
    # save original multi-coil k-space and the sensitivity maps
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with h5py.File(out_path, 'w') as f:
        f.create_dataset('kspace', data=kspace_all, dtype=kspace_all.dtype, compression=None)
        f.create_dataset('csm', data=csm, dtype=np.complex64, compression=None)

    print(f"→ Saved multi-coil k-space and CSM to {out_path}")

def main():
    p = argparse.ArgumentParser(description="Combine multi-coil k-space into single coil via ESPIRiT")
    p.add_argument('-i','--input_folder',  required=True, help="Folder with .h5 multi-coil files")
    p.add_argument('-o','--output_folder', required=True, help="Where to save single-coil .h5 files")
    args = p.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)
    files = sorted(glob.glob(os.path.join(args.input_folder, '*.h5')))
    # Group files by contrast prefix (stem minus last character)
    groups = {}
    for fn in files:
        stem = os.path.splitext(os.path.basename(fn))[0]
        key = stem[:-1]
        groups.setdefault(key, []).append(fn)
    # Process each group with shared normalization
    for group_files in groups.values():
        norm = compute_norm_factor(group_files)
        for fn in sorted(group_files):
            out_fn = os.path.join(args.output_folder, os.path.basename(fn))
            process_file(fn, out_fn, norm)

if __name__ == "__main__":
    main()