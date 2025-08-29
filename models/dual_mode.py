import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any


class DualModeWrapper(nn.Module):
    """Wrapper that adapts existing models for dual-mode operation (signal vs rate estimation)."""
    
    def __init__(self, base_model: nn.Module, task_mode: str = "signal", input_size: int = 512):
        super().__init__()
        self.base_model = base_model
        self.task_mode = task_mode
        self.input_size = input_size
        
        if task_mode == "rate":
            # For rate estimation, we need a regression head
            # First, we need to determine the output size of the base model
            # Let's assume the base model outputs signal-length predictions
            
            self.global_pool = nn.AdaptiveAvgPool1d(1)  # Pool to single value
            self.rate_head = nn.Sequential(
                nn.Linear(1, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 1)  # Single rate value
            )
    
    def forward(self, x):
        # Get base model output
        base_output = self.base_model(x)
        
        if self.task_mode == "signal":
            # For signal estimation, return base model output directly
            return base_output
        else:
            # print("wowwwwww where am i????")
            
            # For rate estimation, pool and predict single rate value
            if len(base_output.shape) == 3:  # (batch, channels, length)
                # print("i'm in batch , channel , length")
                pooled = self.global_pool(base_output).squeeze(-1)  # (batch, channels)
                if pooled.shape[-1] > 1:
                    pooled = pooled.mean(dim=-1, keepdim=True)  # Average across channels
            else:
                # print("i'm in elseee")
                pooled = base_output.mean(dim=-1, keepdim=True)
            
            rate_pred = self.rate_head(pooled)  # (batch, 1)
            # print("rate pred is..",rate_pred)
            
            return rate_pred.squeeze(-1)  # (batch,)


class DualModeModel(nn.Module):
    """A model architecture designed specifically for dual-mode operation."""
    
    def __init__(self, input_size: int, hidden_size: int = 256, num_layers: int = 4, 
                 dropout: float = 0.1, task_mode: str = "signal"):
        super().__init__()
        self.task_mode = task_mode
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Shared feature extraction backbone
        self.backbone = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Task-specific heads
        if task_mode == "signal":
            # Signal reconstruction head
            self.signal_head = nn.Sequential(
                nn.Conv1d(hidden_size * 2, 128, kernel_size=3, padding=1),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Conv1d(128, 64, kernel_size=3, padding=1),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Conv1d(64, 1, kernel_size=1)  # Output single channel
            )
        else:
            # Rate regression head
            self.global_pool = nn.AdaptiveAvgPool1d(1)
            self.rate_head = nn.Sequential(
                nn.Linear(hidden_size * 2, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, 1)
            )
    
    def forward(self, x):
        batch_size, channels, seq_len = x.shape
        
        # Feature extraction
        features = self.backbone(x)  # (batch, 256, seq_len)
        
        # LSTM processing
        features = features.transpose(1, 2)  # (batch, seq_len, 256)
        lstm_out, _ = self.lstm(features)  # (batch, seq_len, hidden_size*2)
        lstm_out = lstm_out.transpose(1, 2)  # (batch, hidden_size*2, seq_len)
        
        if self.task_mode == "signal":
            # Signal reconstruction
            output = self.signal_head(lstm_out)  # (batch, 1, seq_len)
            return output
        else:
            # Rate estimation
            pooled = self.global_pool(lstm_out).squeeze(-1)  # (batch, hidden_size*2)
            rate = self.rate_head(pooled)  # (batch, 1)
            return rate.squeeze(-1)  # (batch,)
