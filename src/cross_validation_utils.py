import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
from typing import Dict, List, Tuple, Optional, Any
import optuna
from optuna.integration import PyTorchLightningPruningCallback
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import os
import pickle
from datetime import datetime
import random


class CrossValidationManager:
    """Enhanced cross-validation manager supporting LOOCV and K-fold CV."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.cv_config = config.get('cross_validation', {})
        
    def create_cross_validation_splits(self, processed_data: Dict[str, Tuple[np.ndarray, np.ndarray]], 
                                     cv_method: str = 'leave_one_out', 
                                     n_folds: int = 5, 
                                     random_state: int = 42,
                                     val_split: float = 0.2) -> List[Dict]:
        """
        Create cross-validation splits with proper subject-wise splitting.
        
        Args:
            processed_data: Dictionary of processed subject data
            cv_method: 'leave_one_out' or 'k_fold'
            n_folds: Number of folds for k-fold CV
            random_state: Random seed for reproducibility
            val_split: Fraction of training subjects to use for validation
            
        Returns:
            List of cross-validation splits with subject-wise validation splits
        """
        subjects = list(processed_data.keys())
        cv_splits = []
        
        if cv_method == 'leave_one_out':
            # Leave-one-out cross-validation with subject-wise validation splits
            for i, test_subject in enumerate(subjects):
                remaining_subjects = [s for s in subjects if s != test_subject]
                
                if len(remaining_subjects) < 2:
                    print(f"Warning: Not enough subjects for proper train/val split in fold {i}")
                    train_subjects = remaining_subjects
                    val_subjects = []
                else:
                    # Set random seed for reproducible splits
                    random.seed(random_state + i)  # Different seed for each fold
                    np.random.seed(random_state + i)
                    
                    n_val_subjects = max(1, int(len(remaining_subjects) * val_split))
                    val_subjects = random.sample(remaining_subjects, n_val_subjects)
                    train_subjects = [s for s in remaining_subjects if s not in val_subjects]
                
                cv_splits.append({
                    'train_subjects': train_subjects,
                    'val_subjects': val_subjects,
                    'test_subject': test_subject,
                    'fold_id': i,
                    'cv_method': 'leave_one_out'
                })
                
        elif cv_method == 'k_fold':
            # K-fold cross-validation with subject-wise validation splits
            np.random.seed(random_state)
            random.seed(random_state)
            subjects_array = np.array(subjects)
            
            # Shuffle subjects
            shuffled_indices = np.random.permutation(len(subjects))
            shuffled_subjects = subjects_array[shuffled_indices]
            
            # Create k-fold splits
            kfold = KFold(n_splits=n_folds, shuffle=False, random_state=None)  # Already shuffled
            
            for fold_id, (train_val_indices, test_indices) in enumerate(kfold.split(shuffled_subjects)):
                train_val_subjects = shuffled_subjects[train_val_indices].tolist()
                test_subjects = shuffled_subjects[test_indices].tolist()
                
                # Further split train_val_subjects into train and validation
                if len(train_val_subjects) < 2:
                    print(f"Warning: Not enough subjects for proper train/val split in fold {fold_id}")
                    train_subjects = train_val_subjects
                    val_subjects = []
                else:
                    # Set random seed for reproducible splits
                    random.seed(random_state + fold_id)
                    np.random.seed(random_state + fold_id)
                    
                    n_val_subjects = max(1, int(len(train_val_subjects) * val_split))
                    val_subjects = random.sample(train_val_subjects, n_val_subjects)
                    train_subjects = [s for s in train_val_subjects if s not in val_subjects]
                
                cv_splits.append({
                    'train_subjects': train_subjects,
                    'val_subjects': val_subjects,
                    'test_subjects': test_subjects,  # Multiple test subjects for k-fold
                    'fold_id': fold_id,
                    'cv_method': 'k_fold',
                    'n_folds': n_folds
                })
                
        else:
            raise ValueError(f"Unsupported CV method: {cv_method}. Use 'leave_one_out' or 'k_fold'")
            
        return cv_splits
    
    def prepare_fold_data_kfold(self, processed_data: Dict[str, Tuple[np.ndarray, np.ndarray]], 
                               train_subjects: List[str], test_subjects: List[str],
                               val_split: float = 0.2) -> Dict:
        """
        Prepare data for k-fold cross-validation (multiple test subjects).
        
        Args:
            processed_data: Dictionary of processed subject data
            train_subjects: List of training subject IDs
            test_subjects: List of test subject IDs
            val_split: Fraction of training data to use for validation
            
        Returns:
            Dictionary containing train, validation, and test data
        """
        # Combine training data from multiple subjects
        train_ppg_list = []
        train_resp_list = []
        
        for subject in train_subjects:
            if subject in processed_data:
                ppg_segments, resp_segments = processed_data[subject]
                train_ppg_list.append(ppg_segments)
                train_resp_list.append(resp_segments)
        
        if not train_ppg_list:
            raise ValueError("No training data available")
        
        train_ppg = np.concatenate(train_ppg_list, axis=0)
        train_resp = np.concatenate(train_resp_list, axis=0)
        
        # Split training data into train and validation
        n_samples = len(train_ppg)
        n_val = int(n_samples * val_split)
        
        # Shuffle indices
        indices = np.random.permutation(n_samples)
        val_indices = indices[:n_val]
        train_indices = indices[n_val:]
        
        # Combine test data from multiple subjects
        test_ppg_list = []
        test_resp_list = []
        
        for subject in test_subjects:
            if subject in processed_data:
                ppg_segments, resp_segments = processed_data[subject]
                test_ppg_list.append(ppg_segments)
                test_resp_list.append(resp_segments)
        
        if not test_ppg_list:
            raise ValueError("No test data available")
            
        test_ppg = np.concatenate(test_ppg_list, axis=0)
        test_resp = np.concatenate(test_resp_list, axis=0)
        
        return {
            'train_ppg': train_ppg[train_indices],
            'train_resp': train_resp[train_indices],
            'val_ppg': train_ppg[val_indices],
            'val_resp': train_resp[val_indices],
            'test_ppg': test_ppg,
            'test_resp': test_resp,
            'test_subjects': test_subjects,
            'train_subjects': train_subjects
        }

    def prepare_subject_wise_fold_data_my_kfold(self, processed_data: Dict[str, Tuple[np.ndarray, np.ndarray]], 
                                      train_subjects: List[str], 
                                      val_subjects: List[str], 
                                      test_subjects: str) -> Dict:
        """
        Prepare data for a specific fold with proper subject-wise splitting.
        
        Args:
            processed_data: Dictionary mapping subject IDs to (PPG, respiratory) data
            train_subjects: List of subjects to use for training
            val_subjects: List of subjects to use for validation
            test_subject: Subject to use for testing
            
        Returns:
            Dictionary containing train/val/test data
        """
        
        # Combine training data from multiple subjects
        train_ppg_list = []
        train_resp_list = []
        
        for subject in train_subjects:
            if subject in processed_data:
                ppg_segments, resp_segments = processed_data[subject]
                train_ppg_list.append(ppg_segments)
                train_resp_list.append(resp_segments)
            else:
                print(f"Warning: Training subject '{subject}' not found in processed data")
        
        if not train_ppg_list:
            raise ValueError("No training data available")
        
        train_ppg = np.concatenate(train_ppg_list, axis=0)
        train_resp = np.concatenate(train_resp_list, axis=0)
        print("train_ppg is ", train_ppg.shape)
        print("train_resp is ", train_resp.shape)
        
        # Combine validation data from multiple subjects
        val_ppg_list = []
        val_resp_list = []
        
        for subject in val_subjects:
            if subject in processed_data:
                ppg_segments, resp_segments = processed_data[subject]
                val_ppg_list.append(ppg_segments)
                val_resp_list.append(resp_segments)
            else:
                print(f"Warning: Validation subject '{subject}' not found in processed data")
        
        if val_ppg_list:
            val_ppg = np.concatenate(val_ppg_list, axis=0)
            val_resp = np.concatenate(val_resp_list, axis=0)
            print("val_ppg is ", val_ppg.shape)
            print("val_resp is ", val_resp.shape)
            print("********* val subjects have been correctly created ********")
            
        else:
            # If no validation subjects, use a portion of training data
            print("Warning: No validation subjects available, using 20% of training data")
            n_samples = len(train_ppg)
            n_val = int(n_samples * 0.2)
            
            # Shuffle indices
            indices = np.random.permutation(n_samples)
            val_indices = indices[:n_val]
            train_indices = indices[n_val:]
            
            val_ppg = train_ppg[val_indices]
            val_resp = train_resp[val_indices]
            train_ppg = train_ppg[train_indices]
            train_resp = train_resp[train_indices]
        
        # Combine testing data from multiple subjects
        test_ppg_list = []
        test_resp_list = []
        
        for subject in test_subjects:
            if subject in processed_data:
                ppg_segments, resp_segments = processed_data[subject]
                test_ppg_list.append(ppg_segments)
                test_resp_list.append(resp_segments)
            else:
                print(f"Warning: Testing subject '{subject}' not found in processed data")
        
        if not test_ppg_list:
            raise ValueError("No testing data available")
        
        test_ppg = np.concatenate(test_ppg_list, axis=0)
        test_resp = np.concatenate(test_resp_list, axis=0)
        print("test_ppg is ", test_ppg.shape)
        print("test_resp is ", test_resp.shape)
        
        return {
            'train_ppg': train_ppg,
            'train_resp': train_resp,
            'val_ppg': val_ppg,
            'val_resp': val_resp,
            'test_ppg': test_ppg,
            'test_resp': test_resp,
            'train_subjects': train_subjects,
            'val_subjects': val_subjects,
            'test_subjects': test_subjects
        } 
    def prepare_subject_wise_fold_data(self, processed_data: Dict[str, Tuple[np.ndarray, np.ndarray]], 
                                      train_subjects: List[str], 
                                      val_subjects: List[str], 
                                      test_subject: str) -> Dict:
        """
        Prepare data for a specific fold with proper subject-wise splitting.
        
        Args:
            processed_data: Dictionary mapping subject IDs to (PPG, respiratory) data
            train_subjects: List of subjects to use for training
            val_subjects: List of subjects to use for validation
            test_subject: Subject to use for testing
            
        Returns:
            Dictionary containing train/val/test data
        """
        
        # Combine training data from multiple subjects
        train_ppg_list = []
        train_resp_list = []
        
        for subject in train_subjects:
            if subject in processed_data:
                ppg_segments, resp_segments = processed_data[subject]
                train_ppg_list.append(ppg_segments)
                train_resp_list.append(resp_segments)
            else:
                print(f"Warning: Training subject '{subject}' not found in processed data")
        
        if not train_ppg_list:
            raise ValueError("No training data available")
        
        train_ppg = np.concatenate(train_ppg_list, axis=0)
        train_resp = np.concatenate(train_resp_list, axis=0)
        
        # Combine validation data from multiple subjects
        val_ppg_list = []
        val_resp_list = []
        
        for subject in val_subjects:
            if subject in processed_data:
                ppg_segments, resp_segments = processed_data[subject]
                val_ppg_list.append(ppg_segments)
                val_resp_list.append(resp_segments)
            else:
                print(f"Warning: Validation subject '{subject}' not found in processed data")
        
        if val_ppg_list:
            val_ppg = np.concatenate(val_ppg_list, axis=0)
            val_resp = np.concatenate(val_resp_list, axis=0)
            print("********* val subjects have been correctly created ********")
        else:
            # If no validation subjects, use a portion of training data
            print("Warning: No validation subjects available, using 20% of training data")
            n_samples = len(train_ppg)
            n_val = int(n_samples * 0.2)
            
            # Shuffle indices
            indices = np.random.permutation(n_samples)
            val_indices = indices[:n_val]
            train_indices = indices[n_val:]
            
            val_ppg = train_ppg[val_indices]
            val_resp = train_resp[val_indices]
            train_ppg = train_ppg[train_indices]
            train_resp = train_resp[train_indices]
        
        # Test data
        if test_subject not in processed_data:
            raise ValueError(f"Test subject '{test_subject}' not found in processed data")
        
        test_ppg, test_resp = processed_data[test_subject]
        
        return {
            'train_ppg': train_ppg,
            'train_resp': train_resp,
            'val_ppg': val_ppg,
            'val_resp': val_resp,
            'test_ppg': test_ppg,
            'test_resp': test_resp,
            'train_subjects': train_subjects,
            'val_subjects': val_subjects,
            'test_subject': test_subject
        }


class OptunaHyperparameterOptimizer:
    """Optuna-based hyperparameter optimization for the PPG respiratory model."""
    
    def __init__(self, config: Dict, processed_data: Dict, cv_splits: List[Dict]):
        self.base_config = config.copy()
        self.processed_data = processed_data
        self.cv_splits = cv_splits
        self.study = None
        
    def suggest_hyperparameters(self, trial: optuna.Trial) -> Dict:
        """
        Suggest hyperparameters for optimization.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Dictionary of suggested hyperparameters
        """
        # Create a copy of the base config
        config = self.base_config.copy()
        
        # Model hyperparameters
        config['model']['hidden_size'] = trial.suggest_categorical('hidden_size', [128, 256, 512])
        config['model']['num_layers'] = trial.suggest_int('num_layers', 2, 8)
        config['model']['dropout'] = trial.suggest_float('dropout', 0.1, 0.5)
        
        # Training hyperparameters
        config['training']['learning_rate'] = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
        config['training']['batch_size'] = trial.suggest_categorical('batch_size', [32, 64, 128])
        config['training']['weight_decay'] = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
        
        # Preprocessing hyperparameters
        config['preprocessing']['bandpass_filter']['low_freq'] = trial.suggest_float('low_freq', 0.01, 0.1)
        config['preprocessing']['bandpass_filter']['high_freq'] = trial.suggest_float('high_freq', 0.6, 4.0)
        config['preprocessing']['normalization'] = trial.suggest_categorical('normalization', 
                                                                           ['z_score', 'min_max', 'robust'])
        
        return config
    
    def objective(self, trial: optuna.Trial) -> float:
        """
        Objective function for Optuna optimization.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Objective value (validation correlation)
        """
        # Get suggested hyperparameters
        config = self.suggest_hyperparameters(trial)
        
        # Import required modules (avoid circular imports)
        from data_utils import prepare_fold_data
        from dataset import create_data_loaders
        from lightning_module import PPGRespiratoryLightningModule
        
        cv_manager = CrossValidationManager(config)
        fold_correlations = []
        
        # Use a subset of folds for faster optimization (e.g., first 3 folds)
        max_folds = min(3, len(self.cv_splits))
        
        for i, cv_split in enumerate(self.cv_splits[:max_folds]):
            try:
                # Prepare fold data
                if cv_split['cv_method'] == 'k_fold':
                    fold_data = cv_manager.prepare_fold_data_kfold(
                        self.processed_data,
                        cv_split['train_subjects'],
                        cv_split['test_subjects'],
                        val_split=config['training']['val_split']
                    )
                else:
                    fold_data = prepare_fold_data(
                        self.processed_data,
                        cv_split['train_subjects'],
                        cv_split['test_subject'],
                        val_split=config['training']['val_split']
                    )
                
                # Create data loaders
                data_loaders = create_data_loaders(
                    fold_data, 
                    batch_size=config['training']['batch_size'],
                    num_workers=2  # Reduced for optimization
                )
                
                # Initialize model
                model = PPGRespiratoryLightningModule(config)
                
                # Setup callbacks with pruning
                callbacks = [
                    PyTorchLightningPruningCallback(trial, monitor="val_correlation"),
                    EarlyStopping(
                        monitor='val_correlation',
                        mode='max',
                        patience=5,  # Reduced patience for faster optimization
                        verbose=False
                    )
                ]
                
                # Setup logger
                logger = TensorBoardLogger(
                    save_dir=os.path.join(config['logging']['log_dir'], 'optuna'),
                    name=f"trial_{trial.number}",
                    version=f'fold_{i}'
                )
                
                # Initialize trainer
                trainer = pl.Trainer(
                    max_epochs=20,  # Reduced epochs for optimization
                    accelerator=config['hardware']['accelerator'],
                    devices=config['hardware']['devices'],
                    precision=config['hardware']['precision'],
                    callbacks=callbacks,
                    logger=logger,
                    enable_progress_bar=False,
                    enable_model_summary=False,
                    enable_checkpointing=False
                )
                
                # Train the model
                trainer.fit(
                    model,
                    train_dataloaders=data_loaders['train'],
                    val_dataloaders=data_loaders['val']
                )
                
                # Get validation correlation
                if trainer.callback_metrics:
                    val_correlation = trainer.callback_metrics.get('val_correlation', 0.0)
                    if isinstance(val_correlation, torch.Tensor):
                        val_correlation = val_correlation.item()
                    fold_correlations.append(val_correlation)
                
            except Exception as e:
                print(f"Error in fold {i}: {e}")
                # Return a poor score if fold fails
                return 0.0
        
        # Return average correlation across folds
        if fold_correlations:
            avg_correlation = np.mean(fold_correlations)
            return avg_correlation
        else:
            return 0.0
    
    def optimize(self, n_trials: int = 50, timeout: Optional[int] = None) -> Dict:
        """
        Run hyperparameter optimization.
        
        Args:
            n_trials: Number of optimization trials
            timeout: Timeout in seconds (optional)
            
        Returns:
            Dictionary containing optimization results
        """
        # Create study
        study_name = f"ppg_respiratory_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.study = optuna.create_study(
            direction='maximize',
            study_name=study_name,
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        )
        
        # Run optimization
        print(f"Starting hyperparameter optimization with {n_trials} trials...")
        self.study.optimize(self.objective, n_trials=n_trials, timeout=timeout)
        
        # Get best parameters
        best_params = self.study.best_params
        best_value = self.study.best_value
        
        print(f"Optimization completed!")
        print(f"Best validation correlation: {best_value:.4f}")
        print(f"Best parameters: {best_params}")
        
        # Create optimized config
        optimized_config = self.base_config.copy()
        
        # Apply best parameters
        for param_name, param_value in best_params.items():
            if param_name == 'hidden_size':
                optimized_config['model']['hidden_size'] = param_value
            elif param_name == 'num_layers':
                optimized_config['model']['num_layers'] = param_value
            elif param_name == 'dropout':
                optimized_config['model']['dropout'] = param_value
            elif param_name == 'learning_rate':
                optimized_config['training']['learning_rate'] = param_value
            elif param_name == 'batch_size':
                optimized_config['training']['batch_size'] = param_value
            elif param_name == 'weight_decay':
                optimized_config['training']['weight_decay'] = param_value
            elif param_name == 'low_freq':
                optimized_config['preprocessing']['bandpass_filter']['low_freq'] = param_value
            elif param_name == 'high_freq':
                optimized_config['preprocessing']['bandpass_filter']['high_freq'] = param_value
            elif param_name == 'normalization':
                optimized_config['preprocessing']['normalization'] = param_value
        
        # Save optimization results
        results = {
            'best_params': best_params,
            'best_value': best_value,
            'study': self.study,
            'optimized_config': optimized_config,
            'n_trials': len(self.study.trials)
        }
        
        # Save to file
        results_dir = os.path.join('results', 'optuna')
        os.makedirs(results_dir, exist_ok=True)
        
        results_path = os.path.join(results_dir, f'{study_name}_results.pkl')
        with open(results_path, 'wb') as f:
            pickle.dump(results, f)
        
        print(f"Optimization results saved to: {results_path}")
        
        return results
