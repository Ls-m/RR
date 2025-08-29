import os
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import pickle
import argparse
from typing import Dict, List
import pandas as pd

from data_utils import DataPreprocessor
from lightning_module import PPGRespiratoryLightningModule


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def load_cv_results(results_path: str) -> Dict:
    """Load cross-validation results."""
    with open(results_path, 'rb') as f:
        cv_results = pickle.load(f)
    return cv_results


def ensemble_predictions(fold_results: List[Dict]) -> Dict:
    """Create ensemble predictions by averaging all fold predictions."""
    
    # Collect all predictions and targets
    all_predictions = []
    all_targets = []
    subject_predictions = {}
    
    for fold_result in fold_results:
        test_subject = fold_result['test_subject']
        predictions = fold_result['predictions']
        targets = fold_result['targets']
        
        # Store subject-specific predictions
        subject_predictions[test_subject] = {
            'predictions': predictions,
            'targets': targets,
            'fold_id': fold_result['fold_id']
        }
        
        # Flatten and collect for overall statistics
        all_predictions.extend(predictions.flatten())
        all_targets.extend(targets.flatten())
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    
    # Calculate overall metrics
    correlation, p_value = pearsonr(all_predictions, all_targets)
    mse = np.mean((all_predictions - all_targets) ** 2)
    mae = np.mean(np.abs(all_predictions - all_targets))
    rmse = np.sqrt(mse)
    
    ensemble_results = {
        'overall_correlation': correlation,
        'overall_p_value': p_value,
        'overall_mse': mse,
        'overall_mae': mae,
        'overall_rmse': rmse,
        'subject_predictions': subject_predictions,
        'all_predictions': all_predictions,
        'all_targets': all_targets
    }
    
    return ensemble_results


def plot_correlation_scatter(predictions: np.ndarray, targets: np.ndarray, 
                           correlation: float, save_path: str):
    """Plot scatter plot of predictions vs targets."""
    
    plt.figure(figsize=(10, 8))
    
    # Create scatter plot
    plt.scatter(targets, predictions, alpha=0.6, s=1)
    
    # Add perfect correlation line
    min_val = min(targets.min(), predictions.min())
    max_val = max(targets.max(), predictions.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Correlation')
    
    # Add best fit line
    z = np.polyfit(targets, predictions, 1)
    p = np.poly1d(z)
    plt.plot(targets, p(targets), 'b-', linewidth=2, label=f'Best Fit (r={correlation:.4f})')
    
    plt.xlabel('Ground Truth Respiratory Signal')
    plt.ylabel('Predicted Respiratory Signal')
    plt.title(f'PPG to Respiratory Waveform Estimation\nPearson Correlation: {correlation:.4f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Make plot square
    plt.axis('equal')
    plt.tight_layout()
    
    # Save plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_sample_predictions(subject_predictions: Dict, num_samples: int, 
                          save_dir: str, sampling_rate: int = 64):
    """Plot random sample predictions."""
    
    subjects = list(subject_predictions.keys())
    selected_subjects = np.random.choice(subjects, min(num_samples, len(subjects)), replace=False)
    
    for i, subject in enumerate(selected_subjects):
        data = subject_predictions[subject]
        predictions = data['predictions']
        targets = data['targets']
        
        # Select a random segment
        if len(predictions) > 0:
            segment_idx = np.random.randint(0, len(predictions))
            pred_segment = predictions[segment_idx, 0, :]  # Remove channel dimension
            target_segment = targets[segment_idx, 0, :]
            
            # Create time axis
            time_axis = np.arange(len(pred_segment)) / sampling_rate
            
            # Calculate correlation for this segment
            segment_corr, _ = pearsonr(pred_segment, target_segment)
            
            plt.figure(figsize=(15, 6))
            
            plt.plot(time_axis, target_segment, label='Ground Truth', linewidth=2, alpha=0.8)
            plt.plot(time_axis, pred_segment, label='Prediction', linewidth=2, alpha=0.8)
            
            plt.xlabel('Time (seconds)')
            plt.ylabel('Normalized Amplitude')
            plt.title(f'Subject {subject} - Segment {segment_idx}\nCorrelation: {segment_corr:.4f}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Save plot
            save_path = os.path.join(save_dir, f'sample_{i+1}_subject_{subject}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()


def plot_subject_correlations(fold_results: List[Dict], save_path: str):
    """Plot correlation for each subject."""
    
    subjects = []
    correlations = []
    
    for fold_result in fold_results:
        test_subject = fold_result['test_subject']
        predictions = fold_result['predictions'].flatten()
        targets = fold_result['targets'].flatten()
        
        if len(predictions) > 1:
            corr, _ = pearsonr(predictions, targets)
            subjects.append(test_subject)
            correlations.append(corr)
    
    # Sort by correlation
    sorted_data = sorted(zip(subjects, correlations), key=lambda x: x[1], reverse=True)
    subjects, correlations = zip(*sorted_data)
    
    plt.figure(figsize=(15, 8))
    bars = plt.bar(range(len(subjects)), correlations, alpha=0.7)
    
    # Color bars based on correlation value
    for i, (bar, corr) in enumerate(zip(bars, correlations)):
        if corr > 0.7:
            bar.set_color('green')
        elif corr > 0.5:
            bar.set_color('orange')
        else:
            bar.set_color('red')
    
    plt.xlabel('Subjects')
    plt.ylabel('Pearson Correlation')
    plt.title('Per-Subject Correlation Results')
    plt.xticks(range(len(subjects)), subjects, rotation=45)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add mean line
    mean_corr = np.mean(correlations)
    plt.axhline(y=mean_corr, color='black', linestyle='--', 
                label=f'Mean: {mean_corr:.4f}')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def generate_results_summary(ensemble_results: Dict, fold_results: List[Dict]) -> str:
    """Generate a comprehensive results summary."""
    
    summary = []
    summary.append("=" * 60)
    summary.append("PPG TO RESPIRATORY WAVEFORM ESTIMATION - RESULTS SUMMARY")
    summary.append("=" * 60)
    summary.append("")
    
    # Overall results
    summary.append("OVERALL RESULTS:")
    summary.append(f"  Pearson Correlation: {ensemble_results['overall_correlation']:.4f}")
    summary.append(f"  P-value: {ensemble_results['overall_p_value']:.2e}")
    summary.append(f"  Mean Squared Error: {ensemble_results['overall_mse']:.4f}")
    summary.append(f"  Mean Absolute Error: {ensemble_results['overall_mae']:.4f}")
    summary.append(f"  Root Mean Squared Error: {ensemble_results['overall_rmse']:.4f}")
    summary.append("")
    
    # Per-fold results
    summary.append("PER-FOLD RESULTS:")
    correlations = []
    losses = []
    maes = []
    
    for fold_result in fold_results:
        test_subject = fold_result['test_subject']
        predictions = fold_result['predictions'].flatten()
        targets = fold_result['targets'].flatten()
        
        if len(predictions) > 1:
            corr, _ = pearsonr(predictions, targets)
            mse = np.mean((predictions - targets) ** 2)
            mae = np.mean(np.abs(predictions - targets))
            
            correlations.append(corr)
            losses.append(mse)
            maes.append(mae)
            
            summary.append(f"  Subject {test_subject}: Corr={corr:.4f}, MSE={mse:.4f}, MAE={mae:.4f}")
    
    summary.append("")
    
    # Statistics
    if correlations:
        summary.append("STATISTICS:")
        summary.append(f"  Mean Correlation: {np.mean(correlations):.4f} ± {np.std(correlations):.4f}")
        summary.append(f"  Mean MSE: {np.mean(losses):.4f} ± {np.std(losses):.4f}")
        summary.append(f"  Mean MAE: {np.mean(maes):.4f} ± {np.std(maes):.4f}")
        summary.append(f"  Best Subject Correlation: {np.max(correlations):.4f}")
        summary.append(f"  Worst Subject Correlation: {np.min(correlations):.4f}")
        summary.append("")
    
    # Model information
    summary.append("MODEL INFORMATION:")
    summary.append(f"  Number of subjects: {len(fold_results)}")
    summary.append(f"  Cross-validation method: Leave-one-out")
    summary.append(f"  Total data points: {len(ensemble_results['all_predictions'])}")
    summary.append("")
    
    summary.append("=" * 60)
    
    return "\n".join(summary)


def main():
    parser = argparse.ArgumentParser(description='Test PPG to Respiratory Waveform Estimation')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--results', type=str, default='results/complete_cv_results.pkl',
                       help='Path to cross-validation results')
    parser.add_argument('--output_dir', type=str, default='results/evaluation',
                       help='Output directory for evaluation results')
    
    args = parser.parse_args()
    
    # Load configuration and results
    config = load_config(args.config)
    cv_results = load_cv_results(args.results)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Evaluating cross-validation results...")
    
    # Get fold results
    fold_results = cv_results['fold_results']
    
    # Create ensemble predictions
    ensemble_results = ensemble_predictions(fold_results)
    
    print(f"Overall Correlation: {ensemble_results['overall_correlation']:.4f}")
    print(f"Overall RMSE: {ensemble_results['overall_rmse']:.4f}")
    
    # Generate plots
    print("Generating plots...")
    
    # Correlation scatter plot
    plot_correlation_scatter(
        ensemble_results['all_predictions'],
        ensemble_results['all_targets'],
        ensemble_results['overall_correlation'],
        os.path.join(args.output_dir, 'correlation_scatter.png')
    )
    
    # Sample predictions
    plot_sample_predictions(
        ensemble_results['subject_predictions'],
        config['evaluation']['plot_samples'],
        args.output_dir,
        config['preprocessing']['downsample']['target_rate']
    )
    
    # Subject correlations
    plot_subject_correlations(
        fold_results,
        os.path.join(args.output_dir, 'subject_correlations.png')
    )
    
    # Generate and save summary
    summary = generate_results_summary(ensemble_results, fold_results)
    
    summary_path = os.path.join(args.output_dir, 'results_summary.txt')
    with open(summary_path, 'w') as f:
        f.write(summary)
    
    print(summary)
    print(f"\nEvaluation completed! Results saved to: {args.output_dir}")
    
    # Save ensemble results
    ensemble_path = os.path.join(args.output_dir, 'ensemble_results.pkl')
    with open(ensemble_path, 'wb') as f:
        pickle.dump(ensemble_results, f)


if __name__ == "__main__":
    main()
