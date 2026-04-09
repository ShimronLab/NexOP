"""
    J-NOP: Joint optimization of NEX and sampling Pattern and reconstruction network

    By Tal Oved
    Primary mail: tal.oved@campus.technion.ac.il    
"""

import os, sys
import torch
from torch import nn
import torch.nn.functional as F


# Was JNOP
class NexOP(nn.Module):
    """
    Joint NEX & sampling Pattern optimization (NexOP).

    Learns `num_masks` independent k-space sampling masks for H×W,
    with optional ACS resampling. Enforces a global acceleration factor R
    by normalizing the expected total number of non-ACS samples across
    all masks to the specified budget, allowing each mask to take a different share.
    ACS lines are always included in the first mask.
    """
    def __init__(
        self,
        image_shape: tuple[int, int],  # (H, W)
        R: float,
        num_masks: int = 3,
        sample_pattern: str = 'horizontal',  # 'horizontal','vertical','3D'
        num_acs_lines: int = 20,
        allow_acs_in_others: bool = True,
        init_method: str = 'random',
        device: torch.device = torch.device('cuda:1')
    ):
        super().__init__()
        self.H, self.W            = image_shape
        self.R                   = R
        self.num_masks           = num_masks
        self.pattern             = sample_pattern
        self.num_acs             = num_acs_lines
        self.allow_acs_in_others = allow_acs_in_others
        self.device              = device

        # 1) full-length and ACS indices
        if self.pattern == 'horizontal':
            length = self.H
            start  = (self.H - self.num_acs)//2
            acs_idx= torch.arange(start, start+self.num_acs, device=device)
        elif self.pattern == 'vertical':
            length = self.W
            start  = (self.W - self.num_acs)//2
            acs_idx= torch.arange(start, start+self.num_acs, device=device)
        else:  # '3D'
            length = self.H * self.W
            sr = (self.H - self.num_acs)//2
            sc = (self.W - self.num_acs)//2
            rows = torch.arange(sr, sr+self.num_acs, device=device)
            cols = torch.arange(sc, sc+self.num_acs, device=device)
            Rg, Cg = torch.meshgrid(rows, cols, indexing='ij')
            acs_idx= (Rg * self.W + Cg).reshape(-1)
        # store
        self.length  = length
        self.acs_idx = acs_idx

        # 2) non-ACS learnable positions
        mask_bool = torch.ones(length, dtype=torch.bool, device=device)
        mask_bool[acs_idx] = False
        nonacs_idx = mask_bool.nonzero(as_tuple=False).squeeze(1)

        # 3) compute global non-ACS budget
        if self.num_masks == 3:
            l = 1
        else:
            l = 1

        if self.pattern in ['horizontal','vertical']:
            tot_nonacs = length / self.R - self.num_acs
        else:
            tot_nonacs = (self.H * self.W) / self.R - (self.num_acs**2)*l # Change back to *1
        self.tot_nonacs = tot_nonacs

        # 4) per-mask insert indices & sizes
        self.insert_idxs = []
        self.sizes       = []
        for i in range(num_masks):
            if i == 0 or not allow_acs_in_others:
                idx = nonacs_idx.clone()
            else:
                idx = torch.arange(length, device=device)
            self.insert_idxs.append(idx)
            self.sizes.append(idx.numel())

        # 5) initialize logits for each mask
        logits = []
        for size in self.sizes:
            if init_method == 'uniform':
                probs = torch.full((size,), 0.5, device=device)
            else:
                probs = torch.rand(size, device=device)
            logits.append(torch.logit(probs, eps=1e-6))
        max_size = max(self.sizes)
        stacked = torch.zeros(num_masks, max_size, device=device)
        for i, lg in enumerate(logits):
            stacked[i, :lg.numel()] = lg
        self.logits = nn.Parameter(stacked)

    def forward(self, tau: float = 0.5) -> torch.Tensor:
        """
        Sample `num_masks` binary masks [num_masks, H, W].
        Expected total non-ACS samples across all masks = budget = tot_nonacs.
        ACS lines are forced in the first mask.
        """
        # continuous probabilities
        p = torch.sigmoid(self.logits)  # [M, max_size]
        # collect learnable p's
        parts = [p[i, :self.sizes[i]] for i in range(self.num_masks)]
        all_p = torch.cat(parts, dim=0)  # [total_positions]

        # global normalization factor
        eps = 1e-6
        r = self.tot_nonacs / (all_p.sum() + eps)

        masks = []
        offset = 0
        for i in range(self.num_masks):
            size = self.sizes[i]
            p_i  = all_p[offset:offset+size] * r
            offset += size
            # Gumbel-softmax sample
            #cat  = torch.stack([1-p_i, p_i], dim=1)  # [size,2]
            cat = torch.stack([1 - p_i, p_i], dim=1).clamp(min=eps)# [size,2]
            samp = F.gumbel_softmax(torch.log(cat), tau=tau, hard=True)[:, 1]

            # place into full-length
            full = torch.zeros(self.length, device=self.device)
            full[self.insert_idxs[i]] = samp
            # force ACS in mask0
            if i == 0:
                full[self.acs_idx] = 1.0

            # reshape to 2D
            if self.pattern == 'horizontal':
                mat = full.unsqueeze(1).repeat(1, self.W)
            elif self.pattern == 'vertical':
                mat = full.unsqueeze(0).repeat(self.H, 1)
            else:
                mat = full.view(self.H, self.W)
            masks.append(mat)

        return torch.stack(masks, dim=0)

    @torch.no_grad()
    def get_prob_masks(self) -> torch.Tensor:
        """
        Continuous probability maps [num_masks, H, W].
        ACS lines set to 1 in the first mask map.
        """
        p = torch.sigmoid(self.logits)
        parts = [p[i, :self.sizes[i]] for i in range(self.num_masks)]
        all_p = torch.cat(parts, dim=0)
        eps = 1e-6
        r = self.tot_nonacs / (all_p.sum() + eps)

        probs = []
        offset = 0
        for i in range(self.num_masks):
            size = self.sizes[i]
            p_i = all_p[offset:offset+size] * r
            offset += size
            full = torch.zeros(self.length, device=self.device)
            full[self.insert_idxs[i]] = p_i
            if i == 0: #or i ==1 or i==2
                full[self.acs_idx] = 1.0
            # reshape
            if self.pattern == 'horizontal':
                mat = full.unsqueeze(1).repeat(1, self.W)
            elif self.pattern == 'vertical':
                mat = full.unsqueeze(0).repeat(self.H, 1)
            else:
                mat = full.view(self.H, self.W)
            probs.append(mat)

        return torch.stack(probs, dim=0)
