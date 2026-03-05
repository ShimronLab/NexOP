import torch
import torch.nn as nn
import math
import numpy as np
import fastmri


def mags_phases_from_pairs(x_2r):
    """
    x_2r: (B, 2R, H, W) where each repetition r has [real, imag] channels
    returns:
      mags:   (B, R, H, W)
      phases: (B, R, H, W)  (atan2)
    """
    real = x_2r[:, 0::2, ...]
    imag = x_2r[:, 1::2, ...]
    mags = torch.sqrt(real**2 + imag**2 + 1e-12)
    phases = torch.atan2(imag, real)
    return mags, phases

def compose_pairs_from_mag_phase(mags, phases):
    real = mags * torch.cos(phases)
    imag = mags * torch.sin(phases)
    B, R, H, W = mags.shape
    pairs = torch.empty(B, 2*R, H, W, device=mags.device, dtype=mags.dtype)
    pairs[:, 0::2, ...] = real           # real_r at even channels
    pairs[:, 1::2, ...] = imag           # imag_r at odd channels
    return pairs
"""
def compose_pairs_from_mag_phase(mags, phases):
    
    mags, phases: (B, R, H, W)
    returns:      (B, 2R, H, W) stacked as [real, imag] per repetition
    
    real = mags * torch.cos(phases)
    imag = mags * torch.sin(phases)
    # interleave [real_r, imag_r] along channel dim
    pairs = torch.stack([real, imag], dim=2)              # (B, R, 2, H, W)
    pairs = pairs.flatten(start_dim=1, end_dim=2)         # (B, 2R, H, W)
    return pairs
"""
def c2r(complex_img, axis=0):
    """
    :input shape: row x col (complex64)
    :output shape: 2 x row x col (float32)
    """
    if isinstance(complex_img, np.ndarray):
        real_img = np.stack((complex_img.real, complex_img.imag), axis=axis)
    elif isinstance(complex_img, torch.Tensor):
        real_img = torch.stack((complex_img.real, complex_img.imag), axis=axis)
    else:
        raise NotImplementedError
    return real_img

def r2c(real_img, axis=0):
    """
    :input shape: 2 x row x col (float32)
    :output shape: row x col (complex64)
    """
    if axis == 0:
        complex_img = torch.complex(real_img[0] ,real_img[1])
    elif axis == 1:
        complex_img = torch.complex(real_img[:,0] , real_img[:,1])
    elif axis == 4:
        complex_img = torch.complex(real_img[:,:,:,:,0] , real_img[:,:,:,:,1])
    elif axis == 3:
        complex_img = torch.complex(real_img[:,:,:,0], real_img[:,:,:,1])
    elif axis == 2:
        complex_img = torch.complex(real_img[:,:,0], real_img[:,:,1])
    else:
        raise NotImplementedError
    return complex_img



#CNN denoiser ======================
def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )

class cnn_denoiser(nn.Module):
    def __init__(self, n_layers,input_size, channels = 64):
        super().__init__()
        layers = []
        layers += conv_block(input_size, channels)

        for _ in range(n_layers-2):
            layers += conv_block(channels, channels)

        layers += nn.Sequential(
            nn.Conv2d(channels, input_size, 3, padding=1),
            nn.BatchNorm2d(input_size)
        )

        self.nw = nn.Sequential(*layers)
        #self.final_block = conv_block(input_size, 2)
        #self.final_norm = nn.BatchNorm2d(2)

    def forward(self, x):
        idt = x # (2, nrow, ncol)
        dw = self.nw(x) + idt # (2, nrow, ncol)
        #dw = self.final_block(dw) # Change from multiple channels to 2 channels output
        #dw = self.final_norm(dw)
         # (2, nrow, ncol)
        return dw

def zdot_reduce_sum(x, y):
    # for (B, H, W) complex -> sum over spatial, keep batch
    return (torch.conj(x) * y).real.sum(dim=(-2, -1))  # shape (B,)


#CG algorithm ======================
class myAtA(nn.Module):
    """
    performs DC step
    """
    def __init__(self, csm, mask, lam):
        super(myAtA, self).__init__()
        self.csm = csm # complex (B x ncoil x nrow x ncol)
        self.mask = mask # complex (B x nrow x ncol)
        self.lam = lam 

    def forward(self, im): #step for batch image
        """
        :im: complex image (B x nrow x nrol)
        """
        #print(f'im shape: {im.shape}')
        #print(f'csm shape: {self.csm.shape}')
        im_coil = self.csm * im.unsqueeze(1) # split coil images (B x ncoil x nrow x ncol)
        #print(f'im_coil shape: {im_coil.shape}')
        im_coil = c2r(im_coil, axis=4)
        k_full = fastmri.fft2c(im_coil) # convert into k-space 
        k_full = r2c(k_full, axis=4) # (B x ncoil x nrow x ncol)
        #print(f'k_full shape: {k_full.shape}')
        #print(f'mask shape: {self.mask.shape}')
        #print(f'kspace shape: {k_full.shape}, mask shape: {self.mask.shape}')
        k_u = k_full * self.mask # undersampling
        k_u = c2r(k_u, axis=4)
        im_u_coil = fastmri.ifft2c(k_u) # convert into image domain
        im_u_coil = r2c(im_u_coil, axis=4) # (B x ncoil x nrow x ncol)
        im_u = torch.sum(im_u_coil * (self.csm.conj()) , axis=1) # coil combine (B x nrow x ncol) * self.csm.conj()
        return im_u + self.lam * im

def myCG(AtA, rhs):
    """
    performs CG algorithm
    :AtA: a class object that contains csm, mask and lambda and operates forward model
    """
    rhs = r2c(rhs, axis=1) # nrow, ncol
    x = torch.zeros_like(rhs)
    i, r, p = 0, rhs, rhs
    rTr = zdot_reduce_sum(r, r)
    while i < 10 and torch.max(rTr) > 1e-10: # 10 iterations or convergence
        Ap = AtA(p)
        alpha = rTr / zdot_reduce_sum(p, Ap) # (B,)
        x = x + alpha[:, None, None] * p
        r = r - alpha[:, None, None] * Ap
        rTrNew = zdot_reduce_sum(r, r)  # (B,)
        beta = rTrNew / rTr
        p = r + beta[:, None, None] * p
        i += 1
        rTr = rTrNew

    return c2r(x, axis=1)

class data_consistency(nn.Module):
    def __init__(self):
        super().__init__()
        self.lam = nn.Parameter(torch.tensor(0.05), requires_grad=True) # 0.05

    def forward(self, z_k, x0, csm, mask):
        rhs = x0 + self.lam * z_k # (2, nrow, ncol)
        AtA = myAtA(csm, mask, self.lam)
        rec = myCG(AtA, rhs)
        return rec

#model =======================    
class MoDL(nn.Module):
    def __init__(self, n_layers, k_iters, input_model='Constant'):
        """
        :n_layers: number of layers
        :k_iters: number of iterations
        """
        super().__init__()
        self.k_iters = k_iters
        self.input_model = input_model
        if input_model == 'Constant':
            input_size = 2
        elif input_model == 'NexOP' or input_model == 'Poisson3' or input_model == 'LOUPE3' or input_model == 'LOUPE3avg' or input_model == 'Poisson3avg':
            input_size = 2 #6
        elif input_model ==  'LOUPE2' or input_model == 'LOUPE2avg':
            input_size = 2
        else:
            input_size = 1
        if input_model == 'NexOP' or input_model == 'Poisson3' or input_model == 'LOUPE3' or input_model == 'LOUPE3avg' or input_model == 'Poisson3avg':
            self.dw = cnn_denoiser(n_layers,3, 64)
            #self.dw1 = cnn_denoiser(n_layers,input_size, 64)
            #self.dw2 = cnn_denoiser(n_layers,input_size, 64)
        elif input_model == 'LOUPE2' or input_model == 'LOUPE2avg':
            self.dw = cnn_denoiser(n_layers,2, 64)
            #self.dw1 = cnn_denoiser(n_layers,input_size, 64) 
        else:
            self.dw = cnn_denoiser(n_layers,input_size, 64)          
        self.dc = data_consistency()

    def forward(self, x0, csm, mask):
        """
        :x0: zero-filled reconstruction (B, 2, nrow, ncol) - float32
        :csm: coil sensitivity map (B, ncoil, nrow, ncol, 2) - float32
        :mask: sampling mask (B, nrow, ncol) - int8
        """
        csm = csm[:,:,:,:,0] + 1j*csm[:,:,:,:,1]
        x_k = x0.clone()
        for k in range(self.k_iters):
            #dw 
            #z_k = self.dw(x_k) # (2 or 6, nrow, ncol)
            
            if self.input_model == 'Constant' or self.input_model == 'Poisson' or self.input_model == 'LOUPE' or self.input_model == 'LOUPE1avg':

                # 1) take magnitudes & phases from current estimate
                mags, phases = mags_phases_from_pairs(x_k)       # (B,3,H,W) each
                # 2) denoise magnitudes only (residual inside cnn_denoiser)
                mags_denoised = torch.nn.functional.softplus(self.dw(mags))       # (B,3,H,W)
                # 3) recompose complex per rep using *previous* phase
                phases = phases.detach()
                z_k = compose_pairs_from_mag_phase(mags_denoised, phases)  # (B,6,H,W)
                #print(f'z_k shape: {z_k.shape}')
                # 4) DC per repetition using that rep's mask and x0 rep
                x_k = self.dc(z_k[:, 0:2, ...], x0[:, 0:2, ...], csm, mask[:, 0, ...])  

            elif self.input_model == 'NexOP' or self.input_model == 'Poisson3' or self.input_model == 'LOUPE3' or self.input_model == 'LOUPE3avg' or self.input_model == 'Poisson3avg':
                # DC for each NEX seperatly
                # ----- 3 reps path -----
                # 1) take magnitudes & phases from current estimate
                mags, phases = mags_phases_from_pairs(x_k)       # (B,3,H,W) each
                # 2) denoise magnitudes only (residual inside cnn_denoiser)
                mags_denoised = torch.nn.functional.softplus(self.dw(mags))       # (B,3,H,W)
                # 3) recompose complex per rep using *previous* phase
                phases = phases.detach()
                z_k = compose_pairs_from_mag_phase(mags_denoised, phases)  # (B,6,H,W)
                # 4) DC per repetition using that rep's mask and x0 rep
                x_k1 = self.dc(z_k[:, 0:2, ...], x0[:, 0:2, ...], csm, mask[:, 0, ...])
                x_k2 = self.dc(z_k[:, 2:4, ...], x0[:, 2:4, ...], csm, mask[:, 1, ...])
                x_k3 = self.dc(z_k[:, 4:6, ...], x0[:, 4:6, ...], csm, mask[:, 2, ...])

                # Concat the results
                x_k = torch.concat((x_k1, x_k2, x_k3), dim=1) # (6, nrow, ncol)
            elif  self.input_model == 'LOUPE2' or self.input_model == 'LOUPE2avg':

                # 1) take magnitudes & phases from current estimate
                mags, phases = mags_phases_from_pairs(x_k)       # (B,3,H,W) each
                # 2) denoise magnitudes only (residual inside cnn_denoiser)
                mags_denoised = torch.nn.functional.softplus(self.dw(mags))       # (B,3,H,W)
                #print(f'mags shape: {mags.shape}, phases shape: {phases.shape}, mags_denoised shape: {mags_denoised.shape}')
                # 3) recompose complex per rep using *previous* phase
                phases = phases.detach()
                z_k = compose_pairs_from_mag_phase(mags_denoised, phases)  # (B,6,H,W)
                # 4) DC per repetition using that rep's mask and x0 rep
                x_k1 = self.dc(z_k[:, 0:2, ...], x0[:, 0:2, ...], csm, mask[:, 0, ...])
                x_k2 = self.dc(z_k[:, 2:4, ...], x0[:, 2:4, ...], csm, mask[:, 1, ...])

                # Concat the results
                x_k = torch.concat((x_k1, x_k2), dim=1) # (6, nrow, ncol)

        return z_k