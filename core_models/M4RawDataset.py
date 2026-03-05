import os
import glob
import h5py
import torch
from torch.utils.data import Dataset
import fastmri
import utils.complex_utils as cplx

def demodulate_linear_phase(img_c, dkx, dky):
    *_, Ny, Nx = img_c.shape
    yy, xx = torch.meshgrid(torch.arange(Ny, device=img_c.device),
                            torch.arange(Nx, device=img_c.device), indexing='ij')
    phase = -2*torch.pi*(dkx*xx/Nx + dky*yy/Ny)  # minus sign = conjugate
    ramp  = torch.exp(1j*phase)
    return  torch.view_as_real(img_c * ramp)

def estimate_shift_from_kspace(img_c):
    K = fastmri.fft2c(img_c)
    mag = K[...,0]**2 + K[...,1]**2  # magnitude squared
    # collapse batch/coil if needed; here assume single image
    _,_,ky, kx = torch.nonzero(mag == mag.max(), as_tuple=True)
    Ny, Nx = mag.shape[-2:]
    center_y, center_x = Ny//2, Nx//2
    dky = (ky[0].item() - center_y)    # integer pixel shift in ky
    dkx = (kx[0].item() - center_x)    # integer pixel shift in kx
    return dkx, dky

def recenter_via_demod(img_c):
    dkx, dky = estimate_shift_from_kspace( img_c)
    # If you need subpixel accuracy, refine dkx/dky (e.g., quadratic fit around the peak)
    return demodulate_linear_phase(torch.view_as_complex(img_c), dkx, dky)

class M4RawDataset(Dataset):
    """
    PyTorch Dataset for the M4Raw denoising task with coil combination.

    - Groups .h5 files by scan ID and modality (e.g. T1, T2, etc.).
    - Each group must have at least `min_reps` repetitions, otherwise it's skipped.
    - Each .h5: k-space shape [S, C, H, W] (slices, coils, height, width).
    - Returns for each (scan, slice):
        reps:   [Nrep, H, W]  - coil-combined magnitude per repetition
        target: [H, W]       - RSS across repetitions
    """
    def __init__(self,
                 root_dir: str,
                 prefix: str,
                 min_reps: int = 3,
                 transform = None,
                 channel_comb: str = 'rss'  # currently only 'rss' supported
                 ):
        super().__init__()
        self.root_dir = root_dir
        self.min_reps = min_reps
        self.channel_comb = channel_comb
        # no transform defult
        self.transform = transform

        # 1) group files by scan_id_modality
        files = glob.glob(os.path.join(root_dir, '*.h5'))
        groups = {}
        for path in files:
            name = os.path.basename(path)
            if not name.endswith('.h5') or f'_{prefix}' not in name:
                continue
            stem = name[:-3]  # remove '.h5'
            # e.g. '2022061203_T101'
            parts = stem.split('_', 1)
            if len(parts) != 2:
                continue
            scan_id, rep_str = parts
            modality = ''.join([c for c in rep_str if not c.isdigit()])
            rep_num = int(''.join([c for c in rep_str if c.isdigit()]))
            key = f"{scan_id}_{modality}"
            groups.setdefault(key, []).append((rep_num, path))

        # 2) filter & sort groups, require >= min_reps
        self.groups = []  # list of lists of paths
        for key, entries in groups.items():
            if len(entries) < self.min_reps:
                continue
            sorted_paths = [p for (_n,p) in sorted(entries, key=lambda x: x[0])]
            self.groups.append(sorted_paths)
        self.groups.sort()

        # 3) probe slice count from first valid group
        if len(self.groups)==0:
            raise RuntimeError('No scans with >= min_reps found.')
        sample_h5 = self.groups[0][0]
        with h5py.File(sample_h5,'r') as hf:
            ks = hf['kspace'][()]  # shape [S,C,H,W]
        self.num_slices = ks.shape[0]

        # 4) build index map: (group_idx, slice_idx)
        self.index_map = []
        for g_idx, paths in enumerate(self.groups):
            for s in range(self.num_slices):
                self.index_map.append((g_idx, s))

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, index):
        g_idx, slice_idx = self.index_map[index]
        paths = self.groups[g_idx]
        rep_imgs = []
        csms = []
        eps = 1e-6  # small value to avoid division by zero
        for p in paths:
            with h5py.File(p,'r') as hf:
                ks_all = hf['kspace'][()]    # [S, C, H, W]
                csm_all = hf['csm'][()]  #  [S, C, H, W]
            #print(f"shape of kspace: {ks_all.shape}, csm: {csm_all.shape}")
            ks = ks_all[slice_idx,:,:,:]           # [C, H, W]
            csm = csm_all[slice_idx,:,:,:]  # [C, H, W]
            ks_t = cplx.to_tensor(ks[:,:,:])  # [H,W,2]  
            csm_t = cplx.to_tensor(csm[:,:,:])
            #ks_t = torch.mean(ks_t, dim=0)  # [H,W,2]
            #### Assumption of single coil ####
            img_c = fastmri.ifft2c(ks_t)      # [H, W, 2]

            # Magnitude of current repetition
            #mag = torch.sqrt(img_c[..., 0]**2 + img_c[..., 1]**2)
            
            #mag_complex = img_c
            # phase-aligned complex image using first repetition's phase
            rep_imgs.append(img_c)
            csms.append(csm_t)  # [C, H, W]
            #reps_mgs_abs.append(mag)  # [H, W]
        
        csm = csms[0]  # use first coil sensitivity map
        reps = torch.stack(rep_imgs, dim=0)
        for r in range(0, reps.shape[0]):
            reps[r,...] = recenter_via_demod(reps[r,...].unsqueeze(0))  # [Nrep,C, H, W]\  
        #reps = recenter_via_demod(reps)
        
        #reps_abs = torch.stack(reps_mgs_abs, dim=0)  # [Nrep, H, W]
        #print(f'target shape: {target.shape}, reps shape: {reps.shape}')
        
        
        
        target = (reps[...])#.mean(dim=0) # [C, H, W]
        #target = reps
        
        #target = reps_abs
        #target = fastmri.complex_abs(target)  # [H, W]

        if self.transform:
            return self.transform(reps, target, csm_t)
        # if no transform, return raw tensors
        return reps, target, csm_t


"""
# Example usage:
if __name__=='__main__':
    
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import fastmri

    ds = M4RawDataset('./M4RawData/multicoil_val', prefix='T1', min_reps=3)
    print('total items:', len(ds))  # = len(groups)*num_slices
    loader = DataLoader(ds, batch_size=1, shuffle=True)
    reps, tgt = next(iter(loader))
    print(reps.shape, tgt.shape)  # [2, Nrep, H, W, B], [B, 2, H, W]
    print()
    # Save one repetition of the first sample as an image
    # reps: [B, Nrep, H, W]
    img = fastmri.complex_abs(reps[0, 0]).cpu().numpy()  # first batch, first repetition, convet to magnitude
    target = tgt[0].cpu().numpy()  # first batch, target image

    plt.figure(figsize=(6,6))
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig('M4RawTestet.png', dpi=150, bbox_inches='tight', pad_inches=0)
    print("Saved image to M4RawTestet.png")

    plt.figure(figsize=(6,6))
    plt.imshow(target, cmap='gray')
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig('M4RawTestet_target.png', dpi=150, bbox_inches='tight', pad_inches=0)
    print("Saved image to M4RawTestet_target.png")
    
"""