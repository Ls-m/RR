import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple


class PPGRespiratoryDataset(Dataset):
    """PyTorch Dataset for PPG to respiratory waveform/rate estimation (dual-mode)."""
    
    def __init__(self, ppg_data: np.ndarray, resp_data: np.ndarray, augment: bool = False, task_mode: str = "signal"):
        """
        Initialize the dataset.
        
        Args:
            ppg_data: PPG signal segments of shape (n_samples, sequence_length)
            resp_data: Respiratory data - either signal segments (n_samples, sequence_length) 
                      or rate values (n_samples, 1) depending on task_mode
            augment: Whether to apply data augmentation (for training only)
            task_mode: "signal" for respiratory signal estimation, "rate" for rate estimation
        """
        # Check for NaN values in input data
        ppg_nan_count = np.isnan(ppg_data).sum()
        resp_nan_count = np.isnan(resp_data).sum()
        
        if ppg_nan_count > 0 or resp_nan_count > 0:
            print(f"Warning: Found {ppg_nan_count} NaN values in PPG data and {resp_nan_count} NaN values in respiratory data")
            # Replace NaN values with 0
            ppg_data = np.nan_to_num(ppg_data, nan=0.0, posinf=0.0, neginf=0.0)
            resp_data = np.nan_to_num(resp_data, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Check for infinite values
        ppg_inf_count = np.isinf(ppg_data).sum()
        resp_inf_count = np.isinf(resp_data).sum()
        
        if ppg_inf_count > 0 or resp_inf_count > 0:
            print(f"Warning: Found {ppg_inf_count} Inf values in PPG data and {resp_inf_count} Inf values in respiratory data")
            # Replace Inf values with 0
            ppg_data = np.nan_to_num(ppg_data, nan=0.0, posinf=0.0, neginf=0.0)
            resp_data = np.nan_to_num(resp_data, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Clip extreme values
        ppg_data = np.clip(ppg_data, -10.0, 10.0)
        
        # Handle clipping based on task mode
        self.task_mode = task_mode
        if task_mode == "signal":
            # For signal estimation, use the existing clipping
            resp_data = np.clip(resp_data, -10.0, 10.0)
        # if task_mode == "rate":
        #     # For rate estimation, clip to physiological range (5-50 BPM)
        #     resp_data = np.clip(resp_data, 5.0, 50.0)
        # else:
        #     # For signal estimation, use the existing clipping
        #     resp_data = np.clip(resp_data, -10.0, 10.0)
        
        self.ppg_data = torch.FloatTensor(ppg_data)
        self.resp_data = torch.FloatTensor(resp_data)
        
        # Add channel dimension for PPG (always 1D CNN input)
        if len(self.ppg_data.shape) == 2:
            self.ppg_data = self.ppg_data.unsqueeze(1)  # (batch, 1, sequence)
        
        # Handle resp_data dimensions based on task mode
        if task_mode == "signal":
            # Signal estimation: add channel dimension like PPG
            if len(self.resp_data.shape) == 2:
                self.resp_data = self.resp_data.unsqueeze(1)  # (batch, 1, sequence)
        else:
            # Rate estimation: keep as (batch, 1) - single value per segment
            if len(self.resp_data.shape) == 2 and self.resp_data.shape[1] == 1:
                self.resp_data = self.resp_data.squeeze(1)  # (batch,)
        
        # Final check for NaN/Inf in tensors
        if torch.isnan(self.ppg_data).any() or torch.isnan(self.resp_data).any():
            print("Error: NaN values still present in tensors after cleaning")
        if torch.isinf(self.ppg_data).any() or torch.isinf(self.resp_data).any():
            print("Error: Inf values still present in tensors after cleaning")
        
        self.augment = augment
        self.max_shift = int(self.ppg_data.shape[-1] * 0.1)  # 10% of sequence length for shifting
    
    def __len__(self) -> int:
        return len(self.ppg_data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        ppg = self.ppg_data[idx]
        resp = self.resp_data[idx]
        
        if self.augment:
            # Apply augmentations with probability 0.5 each
            if torch.rand(1) < 0.5:
                # Add Gaussian noise to PPG only
                noise_std = 0.05 * torch.std(ppg)
                noise = torch.normal(mean=0.0, std=noise_std, size=ppg.shape)
                ppg = ppg + noise
            
            if torch.rand(1) < 0.5:
                # Random amplitude scaling to PPG
                scale_factor = torch.FloatTensor(1).uniform_(0.8, 1.2)
                ppg = ppg * scale_factor
            
            # Only apply temporal augmentations for signal mode
            if self.task_mode == "signal":
                if torch.rand(1) < 0.5:
                    # Random time shift (roll) to both
                    shift = torch.randint(-self.max_shift, self.max_shift + 1, (1,)).item()
                    ppg = torch.roll(ppg, shifts=shift, dims=-1)
                    resp = torch.roll(resp, shifts=shift, dims=-1)
                
                if torch.rand(1) < 0.5:
                    # Random flip (time reversal) to both
                    ppg = torch.flip(ppg, dims=[-1])
                    resp = torch.flip(resp, dims=[-1])
            
            # For rate mode, we don't augment the target rate value
        
        return ppg, resp


def create_data_loaders(fold_data: dict, batch_size: int, num_workers: int = 4, task_mode: str = "signal") -> dict:
    """Create PyTorch DataLoaders for train, validation, and test sets (dual-mode support)."""
    # print("in create data loaders ", fold_data['train_resp'])
    # Create datasets
    train_dataset = PPGRespiratoryDataset(
        fold_data['train_ppg'], 
        fold_data['train_resp'],
        augment=True,  # Enable augmentation for training
        task_mode=task_mode
    )
    
    val_dataset = PPGRespiratoryDataset(
        fold_data['val_ppg'], 
        fold_data['val_resp'],
        augment=False,  # No augmentation for validation
        task_mode=task_mode
    )
    
    test_dataset = PPGRespiratoryDataset(
        fold_data['test_ppg'], 
        fold_data['test_resp'],
        augment=False,  # No augmentation for testing
        task_mode=task_mode
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }







# import torch
# from torch.utils.data import Dataset, DataLoader
# import numpy as np
# from typing import Tuple


# class PPGRespiratoryDataset(Dataset):
#     """PyTorch Dataset for PPG to respiratory waveform estimation."""
    
#     def __init__(self, ppg_data: np.ndarray, resp_data: np.ndarray):
#         """
#         Initialize the dataset.
        
#         Args:
#             ppg_data: PPG signal segments of shape (n_samples, sequence_length)
#             resp_data: Respiratory signal segments of shape (n_samples, sequence_length)
#         """
#         # Check for NaN values in input data
#         ppg_nan_count = np.isnan(ppg_data).sum()
#         resp_nan_count = np.isnan(resp_data).sum()
        
#         if ppg_nan_count > 0 or resp_nan_count > 0:
#             print(f"Warning: Found {ppg_nan_count} NaN values in PPG data and {resp_nan_count} NaN values in respiratory data")
#             # Replace NaN values with 0
#             ppg_data = np.nan_to_num(ppg_data, nan=0.0, posinf=0.0, neginf=0.0)
#             resp_data = np.nan_to_num(resp_data, nan=0.0, posinf=0.0, neginf=0.0)
        
#         # Check for infinite values
#         ppg_inf_count = np.isinf(ppg_data).sum()
#         resp_inf_count = np.isinf(resp_data).sum()
        
#         if ppg_inf_count > 0 or resp_inf_count > 0:
#             print(f"Warning: Found {ppg_inf_count} Inf values in PPG data and {resp_inf_count} Inf values in respiratory data")
#             # Replace Inf values with 0
#             ppg_data = np.nan_to_num(ppg_data, nan=0.0, posinf=0.0, neginf=0.0)
#             resp_data = np.nan_to_num(resp_data, nan=0.0, posinf=0.0, neginf=0.0)
        
#         # Clip extreme values
#         ppg_data = np.clip(ppg_data, -10.0, 10.0)
#         resp_data = np.clip(resp_data, -10.0, 10.0)
        
#         self.ppg_data = torch.FloatTensor(ppg_data)
#         self.resp_data = torch.FloatTensor(resp_data)
        
#         # Add channel dimension for 1D CNN
#         if len(self.ppg_data.shape) == 2:
#             self.ppg_data = self.ppg_data.unsqueeze(1)  # (batch, 1, sequence)
#         if len(self.resp_data.shape) == 2:
#             self.resp_data = self.resp_data.unsqueeze(1)  # (batch, 1, sequence)
        
#         # Final check for NaN/Inf in tensors
#         if torch.isnan(self.ppg_data).any() or torch.isnan(self.resp_data).any():
#             print("Error: NaN values still present in tensors after cleaning")
#         if torch.isinf(self.ppg_data).any() or torch.isinf(self.resp_data).any():
#             print("Error: Inf values still present in tensors after cleaning")
    
#     def __len__(self) -> int:
#         return len(self.ppg_data)
    
#     def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
#         return self.ppg_data[idx], self.resp_data[idx]


# def create_data_loaders(fold_data: dict, batch_size: int, num_workers: int = 4) -> dict:
#     """Create PyTorch DataLoaders for train, validation, and test sets."""
    
#     # Create datasets
#     train_dataset = PPGRespiratoryDataset(
#         fold_data['train_ppg'], 
#         fold_data['train_resp']
#     )
    
#     val_dataset = PPGRespiratoryDataset(
#         fold_data['val_ppg'], 
#         fold_data['val_resp']
#     )
    
#     test_dataset = PPGRespiratoryDataset(
#         fold_data['test_ppg'], 
#         fold_data['test_resp']
#     )
    
#     # Create data loaders
#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=batch_size,
#         shuffle=True,
#         num_workers=num_workers,
#         pin_memory=True,
#         drop_last=True
#     )
    
#     val_loader = DataLoader(
#         val_dataset,
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=num_workers,
#         pin_memory=True
#     )
    
#     test_loader = DataLoader(
#         test_dataset,
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=num_workers,
#         pin_memory=True
#     )
    
#     return {
#         'train': train_loader,
#         'val': val_loader,
#         'test': test_loader
#     }
