import os
import yaml
import torch
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import argparse
from typing import Dict, List, Any
import pickle
from time import sleep
from data_utils import DataPreprocessor, prepare_fold_data
from dataset import create_data_loaders
from lightning_module import PPGRespiratoryLightningModule
from preprocessing_config import PreprocessingConfigManager, EnhancedDataPreprocessor
from cross_validation_utils import CrossValidationManager, OptunaHyperparameterOptimizer
import copy


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def parse_config_override(override_str: str) -> tuple:
    """Parse a config override string in the format 'key.subkey=value'."""
    if '=' not in override_str:
        raise ValueError(f"Invalid override format: {override_str}. Expected 'key=value' or 'key.subkey=value'")
    
    key_path, value_str = override_str.split('=', 1)
    key_list = key_path.split('.')
    
    # Try to convert value to appropriate type
    value = convert_string_to_type(value_str)
    
    return key_list, value


def convert_string_to_type(value_str: str) -> Any:
    """Convert string value to appropriate Python type."""
    # Handle boolean values
    if value_str.lower() in ['true', 'false']:
        return value_str.lower() == 'true'
    
    # Handle None/null values
    if value_str.lower() in ['none', 'null']:
        return None
    
    # Try to convert to int
    try:
        if '.' not in value_str:
            return int(value_str)
    except ValueError:
        pass
    
    # Try to convert to float
    try:
        return float(value_str)
    except ValueError:
        pass
    
    # Handle lists (comma-separated values in brackets)
    if value_str.startswith('[') and value_str.endswith(']'):
        list_str = value_str[1:-1].strip()
        if not list_str:
            return []
        items = [convert_string_to_type(item.strip()) for item in list_str.split(',')]
        return items
    
    # Return as string if no other type matches
    return value_str


def update_nested_dict(config: Dict, key_path: List[str], value: Any) -> None:
    """Update a nested dictionary with a value at the specified key path."""
    current = config
    
    # Navigate to the parent of the target key
    for key in key_path[:-1]:
        if key not in current:
            current[key] = {}
        elif not isinstance(current[key], dict):
            raise ValueError(f"Cannot override {'.'.join(key_path)}: {key} is not a dictionary")
        current = current[key]
    
    # Set the final value
    final_key = key_path[-1]
    old_value = current.get(final_key, "NOT_SET")
    current[final_key] = value
    
    print(f"Config override: {'.'.join(key_path)} = {value} (was: {old_value})")


def apply_config_overrides(config: Dict, overrides: List[str]) -> Dict:
    """Apply command line config overrides to the configuration dictionary."""
    for override in overrides:
        try:
            key_path, value = parse_config_override(override)
            update_nested_dict(config, key_path, value)
        except Exception as e:
            print(f"Warning: Failed to apply override '{override}': {e}")
            continue
    
    return config


def setup_callbacks(config: Dict) -> List:
    """Setup PyTorch Lightning callbacks."""
    callbacks = []
    task_mode = config.get('task', {}).get('mode', 'signal')
    checkpoint_monitor = 'val_correlation'
    check_mode = 'max'
    if task_mode == "rate":
        # Additional callbacks or modifications for rate prediction
        checkpoint_monitor = 'val_loss'
        check_mode = 'min'
    # Model checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(config['logging']['log_dir'], 'checkpoints'),
        filename='{epoch}-{val_loss:.4f}-{val_correlation:.4f}',
        monitor=checkpoint_monitor,
        mode=check_mode,
        save_top_k=config['logging']['save_top_k'],
        save_last=True,
        verbose=True
    )
    callbacks.append(checkpoint_callback)
    
    # Early stopping callback
    early_stopping = EarlyStopping(
        monitor=checkpoint_monitor,
        mode=check_mode,
        patience=config['training']['patience'],
        verbose=True,
        min_delta=0.001
    )
    callbacks.append(early_stopping)
    
    return callbacks


def train_single_fold_enhanced(config: Dict, fold_data: Dict, fold_id: int, 
                              test_info: str, cv_method: str = 'leave_one_out') -> Dict:
    """Train a single fold of cross-validation with enhanced support for k-fold."""
    
    print(f"\n{'='*50}")
    if cv_method == 'k_fold':
        print(f"Training Fold {fold_id} - Test Subjects: {test_info}")
    else:
        print(f"Training Fold {fold_id} - Test Subject: {test_info}")
    print(f"{'='*50}")
    
    fold_config = copy.deepcopy(config)
    if config['model']['pass_fold']:
        fold_config['fold_id'] = fold_id  # Now accessible downstream
    # Create data loaders with task mode support
    task_mode = config.get('task', {}).get('mode', 'signal')
    # print("in train_single fold function the fold_data['train resp'] is ", fold_data['train_resp'])
    data_loaders = create_data_loaders(
        fold_data, 
        batch_size=config['training']['batch_size'],
        num_workers=16,
        task_mode=task_mode
    )
    
    print(f"Train samples: {len(data_loaders['train'].dataset)}")
    print(f"Val samples: {len(data_loaders['val'].dataset)}")
    print(f"Test samples: {len(data_loaders['test'].dataset)}")
    
    
    # Initialize model
    model = PPGRespiratoryLightningModule(fold_config)
    
    # Setup logger
    logger = TensorBoardLogger(
        save_dir=config['logging']['log_dir'],
        name=config['logging']['experiment_name'],
        version=f'fold_{fold_id}_{cv_method}'
    )
    
    # Setup callbacks
    callbacks = setup_callbacks(config)
    
    # Initialize trainer
    gradient_clip_val = config['training'].get('gradient_clip_val', 1.0)
    
    trainer = pl.Trainer(
        max_epochs=config['training']['max_epochs'],
        accelerator=config['hardware']['accelerator'],
        # strategy='ddp_find_unused_parameters_true',
        devices=config['hardware']['devices'],
        precision=config['hardware']['precision'],
        callbacks=callbacks,
        logger=logger,
        enable_progress_bar=True,
        log_every_n_steps=50,
        gradient_clip_val=gradient_clip_val,
        accumulate_grad_batches=1,
        #deterministic=True,
        benchmark=False
    )
    # inputs, targets = next(iter(data_loaders['train']))
    # print("train target in train single fold is ",targets)
    # Train the model
    trainer.fit(
        model,
        train_dataloaders=data_loaders['train'],
        val_dataloaders=data_loaders['val']
    )

    sleep(30)
    
    # Test the model
    test_results = trainer.test(
        model,
        dataloaders=data_loaders['test'],
        ckpt_path='best'
    )
    
      # Brief pause for system stability
    
    # Get predictions for ensemble
    model = PPGRespiratoryLightningModule.load_from_checkpoint(
        trainer.checkpoint_callback.best_model_path,
        config=config
    )
    model.eval()
    
    predictions = trainer.predict(
        model,
        dataloaders=data_loaders['test']
    )
    
    # Combine predictions
    all_predictions = torch.cat([p['predictions'] for p in predictions], dim=0)
    all_targets = torch.cat([p['targets'] for p in predictions], dim=0)
    if task_mode=="signal":
        predictions_sq = all_predictions.squeeze(1).numpy()
        targets_sq = all_targets.squeeze(1).numpy()
    else:
        predictions_sq = all_predictions.numpy()
        targets_sq = all_targets.numpy()

    
    fold_results = {
        'fold_id': fold_id,
        'test_info': test_info,
        'cv_method': cv_method,
        'test_metrics': test_results[0] if test_results else {},
        'predictions': all_predictions.cpu().numpy(),
        'targets': all_targets.cpu().numpy(),
        'best_model_path': trainer.checkpoint_callback.best_model_path
    }
    
    print(f"Fold {fold_id} completed!")
    if test_results:
        print(f"Test Correlation: {test_results[0].get('test_correlation', 'N/A'):.4f}")
        print(f"Test Loss: {test_results[0].get('test_loss', 'N/A'):.4f}")
        print(f"Test MAE: {test_results[0].get('test_mae', 'N/A'):.4f}")
    print('best model path: ',trainer.checkpoint_callback.best_model_path)
    return fold_results


def run_enhanced_cross_validation(config: Dict, cv_method: str = 'leave_one_out', 
                                 n_folds: int = 5, target_subject: str = None,
                                 optimize_hyperparams: bool = False) -> Dict:
    """Run enhanced cross-validation with support for different CV methods."""
    
    print("Starting data preprocessing...")
    
    # Initialize enhanced data preprocessor with config manager
    config_manager = PreprocessingConfigManager()
    enhanced_preprocessor = EnhancedDataPreprocessor(config, config_manager)
    
    # Use regular preprocessor for actual processing (for compatibility)
    preprocessor = DataPreprocessor(config)
    
    # Prepare dataset with dual-mode support
    task_mode = config.get('task', {}).get('mode', 'signal')
    processed_data = preprocessor.prepare_dual_mode_dataset(config['data']['csv_folder'], task_mode)
    
    print(f"Processed data for {len(processed_data)} subjects")
    
    # List available subjects
    available_subjects = list(processed_data.keys())
    print(f"Available subjects: {available_subjects}")
    
    # Save preprocessing configuration
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = config['model']['name']
    config_name = f"{model_name}_preprocessing_{timestamp}"
    
    # Create some basic stats for saving
    total_segments = sum(len(ppg_segments) for ppg_segments, _ in processed_data.values())
    preprocessing_stats = {
        'subjects_processed': len(processed_data),
        'total_segments': total_segments,
        'model_used': model_name,
        'timestamp': timestamp
    }
    
    saved_config_path = config_manager.save_preprocessing_config(
        config, preprocessing_stats, config_name
    )
    print(f"Preprocessing configuration saved as: {config_name}")
    
    # Initialize cross-validation manager
    cv_manager = CrossValidationManager(config)
    
    # Handle single subject training
    if target_subject:
        if target_subject not in available_subjects:
            print(f"Error: Subject '{target_subject}' not found in dataset.")
            print(f"Available subjects: {available_subjects}")
            raise ValueError(f"Subject '{target_subject}' not found in dataset")
        
        print(f"Running single fold training with test subject: {target_subject}")
        train_subjects = [s for s in available_subjects if s != target_subject]
        cv_splits = [{
            'train_subjects': train_subjects,
            'test_subject': target_subject,
            'fold_id': 0,
            'cv_method': 'single_subject'
        }]
    else:
        # print("cv_method is ", cv_method," n_folds is ",n_folds," val split is ",config['training']['val_split'])
        
        # Create cross-validation splits using the enhanced manager with subject-wise validation
        cv_splits = cv_manager.create_cross_validation_splits(
            processed_data, cv_method=cv_method, n_folds=n_folds, 
            random_state=42, val_split=config['training']['val_split']
        )
        print(f"Running {cv_method} cross-validation with {len(cv_splits)} folds")
        print("Using subject-wise validation splitting")
    
    # Hyperparameter optimization (optional)
    optimized_config = config
    if optimize_hyperparams and not target_subject:
        print("Starting hyperparameter optimization...")
        optimizer = OptunaHyperparameterOptimizer(config, processed_data, cv_splits)
        optimization_results = optimizer.optimize(n_trials=10)  # Reduced for demo
        optimized_config = optimization_results['optimized_config']
        print("Using optimized hyperparameters for training")
    
    # Store all fold results
    all_fold_results = []
    
    # Run each fold
    for cv_split in cv_splits:
        fold_id = cv_split['fold_id']
        cv_method_used = cv_split['cv_method']
        
        if cv_method_used == 'k_fold':
            train_subjects = cv_split['train_subjects']
            val_subjects = cv_split['val_subjects']
            test_subjects = cv_split['test_subjects']
            test_info = f"{len(test_subjects)} subjects"
            
            print(f"\nFold {fold_id}: Test subjects = {test_subjects}")
            print(f"Training subjects ({len(train_subjects)}): {train_subjects}")
            print(f"Validation subjects ({len(val_subjects)}): {val_subjects}")
            
            if val_subjects:
                print("**************** got here ****************")
                
                fold_data = cv_manager.prepare_subject_wise_fold_data_my_kfold(
                    processed_data,
                    train_subjects,
                    val_subjects,
                    test_subjects
                )
            else:
                # Fallback to old method if no validation subjects
                fold_data = cv_manager.prepare_fold_data_kfold(
                    processed_data,
                    train_subjects,
                    test_subjects,
                    val_split=optimized_config['training']['val_split']
                )
          
        else:
            
            # LOOCV or single subject with subject-wise validation
            train_subjects = cv_split['train_subjects']
            val_subjects = cv_split.get('val_subjects', [])
            test_subject = cv_split['test_subject']
            test_info = test_subject
            
            print(f"\nFold {fold_id}: Test subject = {test_subject}")
            print(f"Training subjects ({len(train_subjects)}): {train_subjects}")
            print(f"Validation subjects ({len(val_subjects)}): {val_subjects}")
            
            # Use subject-wise fold data preparation
            if val_subjects:
                fold_data = cv_manager.prepare_subject_wise_fold_data(
                    processed_data,
                    train_subjects,
                    val_subjects,
                    test_subject
                )
            else:
                # Fallback to old method if no validation subjects
                fold_data = prepare_fold_data(
                    processed_data,
                    train_subjects,
                    test_subject,
                    val_split=optimized_config['training']['val_split']
                )
        print("fold data is ", fold_data['train_resp'])
        # Train fold
        fold_results = train_single_fold_enhanced(
            optimized_config, fold_data, fold_id, test_info, cv_method_used
        )
        
        all_fold_results.append(fold_results)
        
        # Save intermediate results
        results_path = os.path.join('results', f'fold_{fold_id}_results.pkl')
        os.makedirs('results', exist_ok=True)
        with open(results_path, 'wb') as f:
            pickle.dump(fold_results, f)
        
        # If running single fold, break after first iteration
        if target_subject:
            break
    
    return {
        'fold_results': all_fold_results,
        'processed_data': processed_data,
        'preprocessing_config_name': config_name,
        'preprocessing_config_path': saved_config_path,
        'target_subject': target_subject,
        'cv_method': cv_method,
        'n_folds': n_folds if cv_method == 'k_fold' else len(available_subjects),
        'hyperparameter_optimization': optimize_hyperparams,
        'optimized_config': optimized_config if optimize_hyperparams else None,
        'mode': 'single_fold' if target_subject else cv_method
    }


def main():
    parser = argparse.ArgumentParser(
        description='Enhanced PPG to Respiratory Waveform Estimation Training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Leave-one-out cross-validation (default)
  python src/train_enhanced.py --config configs/improved_config.yaml
  
  # 5-fold cross-validation
  python src/train_enhanced.py --config configs/improved_config.yaml --cv-method k_fold --n-folds 5
  
  # Single fold with specific test subject
  python src/train_enhanced.py --config configs/improved_config.yaml --fold subject_01
  
  # Use capno dataset
  python src/train_enhanced.py --config configs/improved_config.yaml --dataset capno
  
  # With hyperparameter optimization
  python src/train_enhanced.py --config configs/improved_config.yaml --optimize-hyperparams
  
  # Generate data distribution report
  python src/train_enhanced.py --config configs/improved_config.yaml --dataset capno --analyze-data-only
  
  # Config overrides
  python src/train_enhanced.py --override training.learning_rate=0.001 --override model.hidden_size=512
        """
    )
    parser.add_argument('--config', type=str, default='configs/improved_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--cv-method', type=str, choices=['leave_one_out', 'k_fold'], 
                       default='leave_one_out',
                       help='Cross-validation method: leave_one_out or k_fold')
    parser.add_argument('--n-folds', type=int, default=5,
                       help='Number of folds for k-fold cross-validation')
    parser.add_argument('--fold', type=str, default=None,
                       help='Specific subject to use as test set (overrides CV method)')
    parser.add_argument('--dataset', type=str, choices=['bidmc', 'csv', 'capno'], default=None,
                       help='Dataset to use: "bidmc", "csv", or "capno"')
    parser.add_argument('--optimize-hyperparams', action='store_true',
                       help='Enable hyperparameter optimization with Optuna')
    parser.add_argument('--list-subjects', action='store_true',
                       help='List available subjects in the dataset and exit')
    parser.add_argument('--task-mode', type=str, choices=['signal', 'rate'], default=None,
                       help='Task mode: "signal" for respiratory signal estimation, "rate" for respiratory rate estimation')
    parser.add_argument('--override', action='append', default=[],
                       help='Override config values. Format: key.subkey=value (can be used multiple times)')
    parser.add_argument('--print-config', action='store_true',
                       help='Print the final configuration and exit without training')
    
    args = parser.parse_args()
    
    # Set random seeds
    pl.seed_everything(args.seed)
    
    # Load configuration
    config = load_config(args.config)
    
    # Handle dataset selection
    if args.dataset:
        if args.dataset == 'bidmc':
            config['data']['csv_folder'] = 'bidmc_data'
            print(f"Using BIDMC dataset: {config['data']['csv_folder']}")
        elif args.dataset == 'csv':
            config['data']['csv_folder'] = 'csv'
            print(f"Using CSV dataset: {config['data']['csv_folder']}")
        elif args.dataset == 'capno':
            config['data']['csv_folder'] = 'capno'
            print(f"Using Capno dataset: {config['data']['csv_folder']}")
    
    # Handle task mode selection
    if args.task_mode:
        if 'task' not in config:
            config['task'] = {}
        config['task']['mode'] = args.task_mode
        print(f"Using task mode: {args.task_mode}")
    
    # Apply config overrides
    if args.override:
        print(f"\nApplying {len(args.override)} config override(s):")
        config = apply_config_overrides(config, args.override)
    
    
    
    
    # If listing subjects, do that and exit
    if args.list_subjects:
        print("Loading dataset to list available subjects...")
        preprocessor = DataPreprocessor(config)
        task_mode = config.get('task', {}).get('mode', 'signal')
        processed_data = preprocessor.prepare_dual_mode_dataset(config['data']['csv_folder'], task_mode)
        available_subjects = list(processed_data.keys())
        
        print(f"\nDataset: {config['data']['csv_folder']}")
        print(f"Available subjects ({len(available_subjects)}):")
        for i, subject in enumerate(sorted(available_subjects), 1):
            print(f"  {i:2d}. {subject}")
        
        print(f"\nTo train with a specific subject as test set, use:")
        print(f"  python src/train_enhanced.py --config {args.config} --fold <subject_name>")
        if args.dataset:
            print(f"  python src/train_enhanced.py --config {args.config} --dataset {args.dataset} --fold <subject_name>")
        return
    
    print("\nFinal Configuration:")
    print(yaml.dump(config, default_flow_style=False))
    
    # If only printing config, exit here
    if args.print_config:
        print("Configuration printed. Exiting without training.")
        return
    
    # Print training mode
    if args.fold:
        print(f"\n🎯 SINGLE FOLD TRAINING MODE")
        print(f"Test Subject: {args.fold}")
        print(f"Dataset: {config['data']['csv_folder']}")
    else:
        if args.cv_method == 'k_fold':
            print(f"\n🔄 {args.n_folds}-FOLD CROSS-VALIDATION MODE")
        else:
            print(f"\n🔄 LEAVE-ONE-OUT CROSS-VALIDATION MODE")
        print(f"Dataset: {config['data']['csv_folder']}")
        
        if args.optimize_hyperparams:
            print("🔧 Hyperparameter optimization ENABLED")
    
    # Create necessary directories
    os.makedirs(config['logging']['log_dir'], exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Run enhanced cross-validation
    cv_results = run_enhanced_cross_validation(
        config, 
        cv_method=args.cv_method,
        n_folds=args.n_folds,
        target_subject=args.fold,
        optimize_hyperparams=args.optimize_hyperparams
    )
    
    # Save complete results
    if args.fold:
        results_filename = f'single_fold_{args.fold}_results.pkl'
    else:
        results_filename = f'{args.cv_method}_cv_results.pkl'
    
    results_path = os.path.join('results', results_filename)
    with open(results_path, 'wb') as f:
        pickle.dump(cv_results, f)
    
    print(f"\nTraining completed!")
    print(f"Mode: {cv_results['mode']}")
    print(f"CV Method: {cv_results['cv_method']}")
    if cv_results['target_subject']:
        print(f"Test Subject: {cv_results['target_subject']}")
    print(f"Results saved to: {results_path}")
    
    # Print summary statistics
    fold_results = cv_results['fold_results']
    correlations = []
    losses = []
    maes = []
    
    for fold_result in fold_results:
        metrics = fold_result['test_metrics']
        if 'test_correlation' in metrics:
            correlations.append(metrics['test_correlation'])
        if 'test_loss' in metrics:
            losses.append(metrics['test_loss'])
        if 'test_mae' in metrics:
            maes.append(metrics['test_mae'])
    
    if correlations:
        print(f"\nSummary Statistics:")
        if len(correlations) == 1:
            print(f"Test Correlation: {correlations[0]:.4f}")
            print(f"Test Loss: {losses[0]:.4f}")
            print(f"Test MAE: {maes[0]:.4f}")
        else:
            print(f"Average Correlation: {np.mean(correlations):.4f} ± {np.std(correlations):.4f}")
            print(f"Average Loss: {np.mean(losses):.4f} ± {np.std(losses):.4f}")
            print(f"Average MAE: {np.mean(maes):.4f} ± {np.std(maes):.4f}")


if __name__ == "__main__":
    main()
