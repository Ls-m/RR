# revised_dual_rwkv.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple



import torch
import torch.nn as nn
import random

import torch
import torch.nn as nn
import random

class SpecAugment(nn.Module):
    """
    Spectrogram augmentation module for frequency and time masking.
    This version is robust to inputs smaller than the mask parameters.
    """
    def __init__(self, freq_mask_param: int, time_mask_param: int, 
                 num_freq_masks: int = 1, num_time_masks: int = 1):
        super(SpecAugment, self).__init__()
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        """
        Apply SpecAugment to a spectrogram.
        """
        augmented_spec = spec.clone()
        _, _, num_freq_bins, num_time_steps = augmented_spec.shape

        # ✅ Apply Frequency Masking Safely
        # Only apply if the dimension is larger than the smallest possible mask.
        if self.freq_mask_param > 0 and num_freq_bins > self.freq_mask_param:
            for _ in range(self.num_freq_masks):
                # Choose a mask width that is no larger than the parameter OR the dimension itself
                f = random.randint(0, self.freq_mask_param)
                # Choose a valid starting point
                f0 = random.randint(0, num_freq_bins - f)
                augmented_spec[:, :, f0:f0 + f, :] = 0

        # ✅ Apply Time Masking Safely
        # Only apply if the dimension is larger than the smallest possible mask.
        if self.time_mask_param > 0 and num_time_steps > self.time_mask_param:
            for _ in range(self.num_time_masks):
                # Choose a mask width that is no larger than the parameter OR the dimension itself
                t = random.randint(0, self.time_mask_param)
                # Choose a valid starting point
                t0 = random.randint(0, num_time_steps - t)
                augmented_spec[:, :, :, t0:t0 + t] = 0
            
        return augmented_spec
    
class RWKVBlock(nn.Module):
    """RWKV block for time series processing."""
    def __init__(self, d_model: int, d_ff: int = None, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff or 4 * d_model
        # Time mixing (attention-like mechanism)
        self.time_mix_k = nn.Parameter(torch.ones(1, 1, d_model))
        self.time_mix_v = nn.Parameter(torch.ones(1, 1, d_model))
        self.time_mix_r = nn.Parameter(torch.ones(1, 1, d_model))
        self.key = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)
        self.receptance = nn.Linear(d_model, d_model, bias=False)
        self.output = nn.Linear(d_model, d_model, bias=False)
        # Channel mixing (FFN-like mechanism)
        self.channel_mix_k = nn.Parameter(torch.ones(1, 1, d_model))
        self.channel_mix_r = nn.Parameter(torch.ones(1, 1, d_model))
        self.key_c = nn.Linear(d_model, self.d_ff, bias=False)
        self.receptance_c = nn.Linear(d_model, d_model, bias=False)
        self.value_c = nn.Linear(self.d_ff, d_model, bias=False)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        # Learnable decay for better stability
        self.time_decay = nn.Parameter(torch.zeros(1, 1, d_model))

    def forward(self, x, state=None):
        B, T, C = x.shape
        # Time mixing
        x_tm = self.ln1(x)
        if state is None:
            # Initialize state for the first step
            state = torch.zeros(B, C, device=x.device, dtype=x.dtype)
        # Shift and mix with previous state
        x_shifted = torch.cat([state.unsqueeze(1), x_tm[:, :-1]], dim=1)
        k = self.key(x_tm * self.time_mix_k + x_shifted * (1 - self.time_mix_k))
        v = self.value(x_tm * self.time_mix_v + x_shifted * (1 - self.time_mix_v))
        r = self.receptance(x_tm * self.time_mix_r + x_shifted * (1 - self.time_mix_r))
        # WKV computation (simplified RWKV attention)
        wkv = self._compute_wkv(k, v, r)
        x = x + self.dropout(self.output(wkv))
        # Channel mixing
        x_cm = self.ln2(x)
        x_shifted_c = torch.cat([state.unsqueeze(1), x_cm[:, :-1]], dim=1)
        k_c = self.key_c(x_cm * self.channel_mix_k + x_shifted_c * (1 - self.channel_mix_k))
        r_c = self.receptance_c(x_cm * self.channel_mix_r + x_shifted_c * (1 - self.channel_mix_r))
        vk = self.value_c(F.relu(k_c) ** 2)
        x = x + self.dropout(torch.sigmoid(r_c) * vk)
        # Update state for next step
        new_state = x[:, -1]
        return x, new_state

    def _compute_wkv(self, k, v, r):
        """
        Improved WKV (weighted key-value) computation with better numerical stability
        and more efficient recurrent processing.
        """
        B, T, C = k.shape
        device = k.device
        dtype = k.dtype
        # Initialize state for recurrent computation
        wkv_state = torch.zeros(B, C, device=device, dtype=dtype)
        wkv_outputs = []
        # Process each time step recurrently for better temporal modeling
        for t in range(T):
            k_t = k[:, t, :] # (B, C)
            v_t = v[:, t, :] # (B, C)
            r_t = r[:, t, :] # (B, C)
            # Improved attention mechanism with exponential decay
            # Use learnable decay parameter for better adaptation
            decay = torch.exp(self.time_decay).squeeze(1)  # (1, C)
            # Update state with exponential moving average
            wkv_state = decay * wkv_state + (1 - decay) * v_t
            # Apply receptance gating with improved activation
            receptance = torch.sigmoid(r_t)
            # Compute output with residual connection for better gradient flow
            wkv_t = receptance * wkv_state + 0.1 * v_t # Small residual connection
            wkv_outputs.append(wkv_t.unsqueeze(1))
        # Concatenate all time steps
        wkv = torch.cat(wkv_outputs, dim=1) # (B, T, C)
        # Apply layer normalization for better stability
        wkv = F.layer_norm(wkv, wkv.shape[-1:])
        return wkv

class RevisedRWKV_DualBranch(nn.Module):
    """RWKV model with dual branches for time and frequency domain processing."""
    def __init__(self, input_size: int, hidden_size: int = 256,
                 num_layers: int = 6, dropout: float = 0.1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # ✅ 1. Initialize the SpecAugment module
        # These parameters are tunable. Start with small values.
        # For example, mask up to 15 freq bins and 20 time steps.
        self.spec_augment = SpecAugment(freq_mask_param=15, 
                                        time_mask_param=20, 
                                        num_freq_masks=2, 
                                        num_time_masks=2)
        # Time domain input projection
        self.time_proj = nn.Sequential(
            nn.Conv1d(1, hidden_size // 2, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden_size // 2),
            nn.ReLU(),
            nn.Conv1d(hidden_size // 2, hidden_size, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        # Time domain RWKV blocks
        self.time_rwkv = nn.ModuleList([
            RWKVBlock(hidden_size, dropout=dropout)
            for _ in range(num_layers)
        ])
        # Frequency domain projection (magnitude and phase as 2 channels, now using Conv2d for STFT)
        self.freq_proj = nn.Sequential(
            nn.Conv2d(2, hidden_size // 2, kernel_size=1),
            nn.BatchNorm2d(hidden_size // 2),
            nn.ReLU(),
            nn.Conv2d(hidden_size // 2, hidden_size, kernel_size=1),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        # Frequency domain RWKV blocks
        self.freq_rwkv = nn.ModuleList([
            RWKVBlock(hidden_size, dropout=dropout)
            for _ in range(num_layers)
        ])
        # Cross-attention for fusion
        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=8, dropout=dropout)
        # Fusion normalization
        self.fusion_norm = nn.LayerNorm(hidden_size)
        # Output projection (same as original RWKV)
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 4, 1)
        )
        self.ln_out = nn.LayerNorm(hidden_size)

    def forward(self, x):
        # x shape: (batch_size, 1, sequence_length)
        B, _, T = x.shape
        # Time domain branch
        time_x = self.time_proj(x) # (B, hidden_size, T)
        time_x = time_x.transpose(1, 2) # (B, T, hidden_size)
        state_time = None
        for rwkv_block in self.time_rwkv:
            time_x, state_time = rwkv_block(time_x, state_time)
        # Frequency domain branch with STFT
        x_squeeze = x.squeeze(1)  # (B, T)
        x_stft_complex = torch.stft(x_squeeze, n_fft=256, hop_length=64, 
                                    win_length=256, window=torch.hann_window(256).to(x.device), 
                                    return_complex=True)
        real_part = x_stft_complex.real
        imag_part = x_stft_complex.imag
        freq_input = torch.stack([real_part, imag_part], dim=1)  # Shape: (B, 2, F, T)
        if self.training:
            freq_input = self.spec_augment(freq_input)
        freq_x = self.freq_proj(freq_input)  # (B, hidden_size, F, Time_windows)
        # Treat as sequence: flatten F and Time_windows into seq dim
        freq_x = freq_x.flatten(2, 3)  # (B, hidden_size, F*Time_windows)
        freq_x = freq_x.transpose(1, 2)  # (B, Seq, hidden_size) where Seq = F*Time_windows
        state_freq = None
        for rwkv_block in self.freq_rwkv:
            freq_x, state_freq = rwkv_block(freq_x, state_freq)
        # Interpolate freq_x to match time dimension T for fusion
        freq_x = F.interpolate(freq_x.transpose(1, 2), size=T, mode='linear').transpose(1, 2)  # (B, T, hidden_size)
        # Fuse with cross-attention: time queries freq
        fused, _ = self.cross_attn(query=time_x, key=freq_x, value=freq_x)
        fused = self.fusion_norm(fused + time_x)  # Residual
        fused = self.ln_out(fused)
        # Output projection
        output = self.output_proj(fused)  # (B, T, 1)
        output = output.transpose(1, 2)  # (B, 1, T)
        return output








