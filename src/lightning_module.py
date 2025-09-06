import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import numpy as np
from scipy.stats import pearsonr
from typing import Dict, Any
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import get_model

# Import AdaBelief optimizer
try:
    from adabelief_pytorch import AdaBelief
    ADABELIEF_AVAILABLE = True
except ImportError:
    ADABELIEF_AVAILABLE = False
    print("Warning: AdaBelief optimizer not available. Install with: pip install adabelief-pytorch")


class PPGRespiratoryLightningModule(pl.LightningModule):
    """PyTorch Lightning module for PPG to respiratory waveform/rate estimation (dual-mode)."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        
        # Extract fold_id from config
        self.fold_id = config.get('fold_id', None)  # Default to None if not present
        # Task mode configuration
        self.task_mode = config.get('task', {}).get('mode', 'signal')
        print(f"Initializing Lightning module in {self.task_mode} mode")
        
        # Model configuration
        model_config = config['model']
        self.model = get_model(
            model_name=model_config['name'],
            task_mode=self.task_mode,
            input_size=model_config['input_size'],
            hidden_size=model_config['hidden_size'],
            num_layers=model_config['num_layers'],
            dropout=model_config['dropout'],
            fold_id=self.fold_id
        )
        
        # Training configuration
        self.learning_rate = config['training']['learning_rate']
        self.optimizer_name = config['training']['optimizer']
        self.scheduler_name = config['training']['scheduler']
        
        # Loss function based on task mode
        if self.task_mode == "rate":
            # For rate estimation, use MAE loss (more robust for regression)
            self.criterion = nn.L1Loss()  # Mean Absolute Error
        else:
            # For signal estimation, use MSE loss
            self.criterion = nn.MSELoss()

        # Metrics storage for epoch-end calculations
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        ppg, resp = batch
        # print("first of all the firsts..",resp)
        # Check for NaN in input data
        if torch.isnan(ppg).any() or torch.isnan(resp).any():
            print(f"NaN detected in batch {batch_idx}")
            print(f"PPG NaN count: {torch.isnan(ppg).sum().item()}")
            print(f"RESP NaN count: {torch.isnan(resp).sum().item()}")
            # Skip this batch
            return None
        
        # Check for infinite values
        if torch.isinf(ppg).any() or torch.isinf(resp).any():
            print(f"Inf detected in batch {batch_idx}")
            return None
        
        # Clamp input values to reasonable range
        ppg = torch.clamp(ppg, min=-10.0, max=10.0)
        if self.task_mode == "signal":
            resp = torch.clamp(resp, min=-10.0, max=10.0)

        # if self.task_mode == "rate":
        #     # For rate targets, clamp to physiological range (5-50 BPM)
        #     resp = torch.clamp(resp, min=5.0, max=50.0)
        # else:
        #     # For signal targets, use existing range
        #     resp = torch.clamp(resp, min=-10.0, max=10.0)
        
        pred = self.forward(ppg)
        
        # Check for NaN in predictions
        if torch.isnan(pred).any():
            print(f"NaN in predictions at batch {batch_idx}")
            print(f"Model parameters have NaN: {any(torch.isnan(p).any() for p in self.parameters())}")
            return None
        
        # Check for infinite predictions
        if torch.isinf(pred).any():
            print(f"Inf in predictions at batch {batch_idx}")
            return None
        
        if self.task_mode == "signal":
            pred = torch.clamp(pred, min=-10.0, max=10.0)

        # # Clamp predictions based on task mode
        # if self.task_mode == "rate":
        #     pred = torch.clamp(pred, min=5.0, max=50.0)
        # else:
        #     pred = torch.clamp(pred, min=-10.0, max=10.0)
        # print("resp is")
        # print(resp)
        # print(pred)
        # exit()
        loss = self.criterion(pred, resp)
        
        # Check for NaN loss
        if torch.isnan(loss):
            print(f"NaN loss at batch {batch_idx}")
            print(f"Pred stats: min={pred.min():.4f}, max={pred.max():.4f}, mean={pred.mean():.4f}, std={pred.std():.4f}")
            print(f"Target stats: min={resp.min():.4f}, max={resp.max():.4f}, mean={resp.mean():.4f}, std={resp.std():.4f}")
            return None
        
        # Check for infinite loss
        if torch.isinf(loss):
            print(f"Inf loss at batch {batch_idx}")
            return None
        
        # Store outputs for epoch-end calculations
        self.training_step_outputs.append({
            'train_loss': loss,
            'pred': pred.detach().cpu(),
            'target': resp.detach().cpu()
        })
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def on_train_epoch_end(self):
        if not self.training_step_outputs:
            return
            
        # Calculate average loss
        avg_loss = torch.stack([x['train_loss'] for x in self.training_step_outputs]).mean()
        
        # Calculate Pearson correlation (adapted for both modes)
        all_preds = torch.cat([x['pred'] for x in self.training_step_outputs], dim=0)
        all_targets = torch.cat([x['target'] for x in self.training_step_outputs], dim=0)
        
        if self.task_mode == "rate":
            # For rate mode, predictions and targets are 1D
            preds_flat = all_preds.numpy()
            targets_flat = all_targets.numpy()
        else:
            # For signal mode, flatten the sequences
            preds_flat = all_preds.flatten().numpy()
            targets_flat = all_targets.flatten().numpy()
        targets_flat = all_targets.flatten().numpy()
        
        # Calculate correlation
        if len(preds_flat) > 1 and np.std(preds_flat) > 0 and np.std(targets_flat) > 0:
            correlation, _ = pearsonr(preds_flat, targets_flat)
        else:
            correlation = 0.0
        
        # Calculate MAE
        mae = F.l1_loss(all_preds, all_targets)
        
        # Log metrics for TensorBoard
        self.log('train_loss_epoch', avg_loss, prog_bar=False)
        self.log('train_correlation', correlation, prog_bar=False)
        self.log('train_mae', mae, prog_bar=False)
        
        # Clear outputs
        self.training_step_outputs.clear()
    
    def validation_step(self, batch, batch_idx):
        ppg, resp = batch
        
        # Clamp inputs based on task mode
        ppg = torch.clamp(ppg, min=-10.0, max=10.0)
        if self.task_mode == "signal":
            resp = torch.clamp(resp, min=-10.0, max=10.0)
        # if self.task_mode == "rate":
        #     resp = torch.clamp(resp, min=5.0, max=50.0)
        # else:
        #     resp = torch.clamp(resp, min=-10.0, max=10.0)
        
        pred = self.forward(ppg)
        
        # Clamp predictions based on task mode
        if self.task_mode == "signal":
            pred = torch.clamp(pred, min=-10.0, max=10.0)
        # if self.task_mode == "rate":
        #     pred = torch.clamp(pred, min=5.0, max=50.0)
        # else:
        #     pred = torch.clamp(pred, min=-10.0, max=10.0)
        
        loss = self.criterion(pred, resp)
        
        # Store outputs for epoch-end calculations
        self.validation_step_outputs.append({
            'val_loss': loss,
            'pred': pred.detach().cpu(),
            'target': resp.detach().cpu()
        })
        
        return loss
    
    def on_validation_epoch_end(self):
        if not self.validation_step_outputs:
            return
            
        # Calculate average loss
        avg_loss = torch.stack([x['val_loss'] for x in self.validation_step_outputs]).mean()
        
        # Calculate Pearson correlation (adapted for both modes)
        all_preds = torch.cat([x['pred'] for x in self.validation_step_outputs], dim=0)
        all_targets = torch.cat([x['target'] for x in self.validation_step_outputs], dim=0)
        
        if self.task_mode == "rate":
            # For rate mode, predictions and targets are 1D
            preds_flat = all_preds.numpy()
            targets_flat = all_targets.numpy()
        else:
            # For signal mode, flatten the sequences
            preds_flat = all_preds.flatten().numpy()
            targets_flat = all_targets.flatten().numpy()

        # Calculate correlation
        if len(preds_flat) > 1 and np.std(preds_flat) > 0 and np.std(targets_flat) > 0:
            correlation, _ = pearsonr(preds_flat, targets_flat)
        else:
            correlation = 0.0
        
        # Calculate MAE
        mae = F.l1_loss(all_preds, all_targets)
        
        # Log metrics
        self.log('val_loss', avg_loss, prog_bar=True)
        self.log('val_correlation', correlation, prog_bar=False)
        self.log('val_mae', mae, prog_bar=False)
        
        # Clear outputs
        self.validation_step_outputs.clear()
    
    def test_step(self, batch, batch_idx):
        ppg, resp = batch
        pred = self.forward(ppg)
        
        loss = self.criterion(pred, resp)
        
        # Store outputs for epoch-end calculations
        self.test_step_outputs.append({
            'test_loss': loss,
            'pred': pred.detach().cpu(),
            'target': resp.detach().cpu()
        })
        
        return loss
    
    def on_test_epoch_end(self):
        if not self.test_step_outputs:
            return
            
        # Calculate average loss
        avg_loss = torch.stack([x['test_loss'] for x in self.test_step_outputs]).mean()
        
        # Calculate Pearson correlation
        all_preds = torch.cat([x['pred'] for x in self.test_step_outputs], dim=0)
        all_targets = torch.cat([x['target'] for x in self.test_step_outputs], dim=0)
        
        # Flatten for correlation calculation
        preds_flat = all_preds.flatten().numpy()
        targets_flat = all_targets.flatten().numpy()
        
        # Calculate correlation
        if len(preds_flat) > 1 and np.std(preds_flat) > 0 and np.std(targets_flat) > 0:
            correlation, _ = pearsonr(preds_flat, targets_flat)
        else:
            correlation = 0.0
        
        # Calculate MAE
        mae = F.l1_loss(all_preds, all_targets)
        print(all_preds)
        print(all_targets)
        # print(mae)
        # exit()
        # Log metrics
        self.log('test_loss', avg_loss)
        self.log('test_correlation', correlation)
        self.log('test_mae', mae)
        
        # Store results for external access
        self.test_results = {
            'predictions': all_preds,
            'targets': all_targets,
            'loss': avg_loss.item(),
            'correlation': correlation,
            'mae': mae.item()
        }
        
        # Clear outputs
        self.test_step_outputs.clear()
    
    def predict_step(self, batch, batch_idx):
        ppg, resp = batch
        pred = self.forward(ppg)
        return {
            'predictions': pred,
            'targets': resp
        }
    
    def configure_optimizers(self):
        # Get optimizer parameters from config
        weight_decay = self.config['training'].get('weight_decay', 1e-4)
        
        # Ensure weight_decay is a valid positive float
        if weight_decay is None or not isinstance(weight_decay, (int, float)) or weight_decay < 0:
            print(f"Warning: Invalid weight_decay value: {weight_decay}. Using default 1e-4")
            weight_decay = 1e-4
        
        weight_decay = float(weight_decay)  # Ensure it's a float
        
        # AdaBelief specific parameters
        adabelief_eps = self.config['training'].get('adabelief_eps', 1e-16)
        adabelief_betas = self.config['training'].get('adabelief_betas', (0.9, 0.999))
        adabelief_weight_decouple = self.config['training'].get('adabelief_weight_decouple', True)
        adabelief_rectify = self.config['training'].get('adabelief_rectify', True)
        
        # Optimizer selection
        if self.optimizer_name.lower() == 'adam':
            optimizer = Adam(self.parameters(), lr=self.learning_rate)
        elif self.optimizer_name.lower() == 'adamw':
            optimizer = AdamW(self.parameters(), lr=self.learning_rate, weight_decay=weight_decay)
        elif self.optimizer_name.lower() == 'sgd':
            optimizer = SGD(self.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=weight_decay)
        elif self.optimizer_name.lower() == 'adabelief':
            if not ADABELIEF_AVAILABLE:
                print("Warning: AdaBelief not available, falling back to AdamW")
                optimizer = AdamW(self.parameters(), lr=self.learning_rate, weight_decay=weight_decay)
            else:
                optimizer = AdaBelief(
                    self.parameters(),
                    lr=self.learning_rate,
                    eps=adabelief_eps,
                    betas=adabelief_betas,
                    weight_decouple=adabelief_weight_decouple,
                    rectify=adabelief_rectify,
                    weight_decay=weight_decay
                )
                print(f"Using AdaBelief optimizer with lr={self.learning_rate}, weight_decay={weight_decay}")
        else:
            print(f"Warning: Unknown optimizer '{self.optimizer_name}', falling back to Adam")
            optimizer = Adam(self.parameters(), lr=self.learning_rate)
        
        # Scheduler configuration
        if self.scheduler_name.lower() == 'reduce_on_plateau':
            scheduler = ReduceLROnPlateau(
                optimizer, 
                mode='min', 
                factor=0.5, 
                patience=3,
                
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'val_loss',
                    'interval': 'epoch',
                    'frequency': 1
                }
            }
        elif self.scheduler_name.lower() == 'cosine':
            scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
            return {
                'optimizer': optimizer,
                'lr_scheduler': scheduler
            }
        else:
            return optimizer
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Log learning rate after each training batch for TensorBoard monitoring."""
        # Get current learning rate from optimizer
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        
        # Log learning rate to TensorBoard
        self.log('learning_rate', current_lr, on_step=True, on_epoch=False, prog_bar=False)
        
        # Log additional optimizer-specific metrics for AdaBelief
        if self.optimizer_name.lower() == 'adabelief' and ADABELIEF_AVAILABLE:
            optimizer = self.trainer.optimizers[0]
            if hasattr(optimizer, 'state') and len(optimizer.state) > 0:
                # Get first parameter's state for monitoring
                first_param = next(iter(self.parameters()))
                if first_param in optimizer.state:
                    state = optimizer.state[first_param]
                    if 'step' in state:
                        self.log('optimizer_step', float(state['step']), on_step=True, on_epoch=False, prog_bar=False)
                    if 'exp_avg' in state and 'exp_avg_sq' in state:
                        # Log gradient statistics
                        exp_avg_norm = state['exp_avg'].norm().item()
                        exp_avg_sq_norm = state['exp_avg_sq'].norm().item()
                        self.log('exp_avg_norm', exp_avg_norm, on_step=True, on_epoch=False, prog_bar=False)
                        self.log('exp_avg_sq_norm', exp_avg_sq_norm, on_step=True, on_epoch=False, prog_bar=False)
