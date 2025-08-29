import os
import pandas as pd
import numpy as np
from scipy import signal
from scipy.stats import zscore
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from typing import Tuple, List, Dict, Optional
import warnings
from sklearn.model_selection import train_test_split
import random
warnings.filterwarnings('ignore')


class SubjectWiseDataPreprocessor:
    """Enhanced data preprocessing utilities with proper subject-wise splitting."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.data_config = config['data']
        self.preprocess_config = config['preprocessing']
        
    def load_csv_files(self, csv_folder: str) -> Dict[str, pd.DataFrame]:
        """Load all CSV files from the specified folder."""
        csv_files = {}
        csv_path = csv_folder
        
        for filename in os.listdir(csv_path):
            if filename.endswith('.csv'):
                subject_id = filename.replace('.csv', '')
                try:
                    df = pd.read_csv(os.path.join(csv_path, filename))
                    
                    # Skip the first row which contains sampling rates (for some datasets)
                    if df.iloc[0].dtype == 'object' or any(df.iloc[0].astype(str).str.contains('Hz|Rate|rate', na=False)):
                        df = df.iloc[1:].reset_index(drop=True)
                    
                    # Convert to numeric
                    for col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    csv_files[subject_id] = df
                    print(f"Loaded {subject_id}: {df.shape}, Columns: {df.columns.tolist()}")
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
                    
        return csv_files
    
    def apply_bandpass_filter(self, signal_data: np.ndarray, 
                            sampling_rate: int) -> np.ndarray:
        """Apply bandpass filter to the signal."""
        low_freq = self.preprocess_config['bandpass_filter']['low_freq']
        high_freq = self.preprocess_config['bandpass_filter']['high_freq']
        order = self.preprocess_config['bandpass_filter']['order']
        
        nyquist = sampling_rate / 2
        low = low_freq / nyquist
        high = high_freq / nyquist
        
        # Validate frequency bounds
        if low <= 0:
            print(f"Warning: Low frequency {low_freq} too low, setting to 0.01 Hz")
            low = 0.01 / nyquist
        if high >= 1:
            print(f"Warning: High frequency {high_freq} too high, setting to {nyquist * 0.9} Hz")
            high = 0.9
        if low >= high:
            print(f"Warning: Low frequency >= High frequency, adjusting...")
            low = high * 0.1
        
        try:
            # Use second-order sections (SOS) for better numerical stability
            sos = signal.butter(order, [low, high], btype='band', output='sos')
            filtered_signal = signal.sosfiltfilt(sos, signal_data)
            
            # Check for NaN values after filtering
            if np.isnan(filtered_signal).any():
                print(f"Warning: NaN values detected after filtering, using original signal")
                return signal_data
            
            return filtered_signal
            
        except Exception as e:
            print(f"Error in bandpass filtering: {e}")
            print("Returning original signal without filtering")
            return signal_data
    
    def downsample_signal(self, signal_data: np.ndarray, 
                         original_rate: int, target_rate: int) -> np.ndarray:
        """Downsample the signal to target sampling rate."""
        if original_rate == target_rate:
            return signal_data
            
        downsample_factor = original_rate // target_rate
        downsampled = signal.decimate(signal_data, downsample_factor, ftype='fir')
        
        return downsampled
    
    def normalize_signal(self, signal_data: np.ndarray, 
                        method: str = 'z_score') -> np.ndarray:
        """Normalize the signal using specified method."""
        # Check for invalid values
        if len(signal_data) == 0 or np.all(np.isnan(signal_data)):
            return signal_data
        
        # Remove outliers using IQR method
        Q1 = np.percentile(signal_data, 25)
        Q3 = np.percentile(signal_data, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR
        
        # Clip outliers
        signal_data = np.clip(signal_data, lower_bound, upper_bound)
            
        if method == 'z_score':
            # Handle case where std is 0
            mean_val = np.mean(signal_data)
            std_val = np.std(signal_data)
            
            if std_val == 0 or np.isnan(std_val) or np.isnan(mean_val):
                return np.zeros_like(signal_data)
            
            normalized = (signal_data - mean_val) / std_val
            
            # Additional clipping to prevent extreme values
            normalized = np.clip(normalized, -5.0, 5.0)
            
            # Replace any remaining NaN/Inf values with 0
            normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
            return normalized
            
        elif method == 'min_max':
            min_val = np.min(signal_data)
            max_val = np.max(signal_data)
            
            if max_val == min_val:
                return np.zeros_like(signal_data)
            
            normalized = 2 * (signal_data - min_val) / (max_val - min_val) - 1
            normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
            return normalized
            
        elif method == 'robust':
            median_val = np.median(signal_data)
            mad = np.median(np.abs(signal_data - median_val))
            
            if mad == 0:
                return np.zeros_like(signal_data)
            
            normalized = (signal_data - median_val) / (1.4826 * mad)
            normalized = np.clip(normalized, -5.0, 5.0)
            normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
            return normalized
        else:
            return signal_data
    
    def segment_signal(self, ppg_signal: np.ndarray, resp_signal: np.ndarray,
                      segment_length: int, overlap: float) -> Tuple[np.ndarray, np.ndarray]:
        """Segment signals into overlapping windows."""
        step_size = int(segment_length * (1 - overlap))
        
        ppg_segments = []
        resp_segments = []
        
        for start in range(0, len(ppg_signal) - segment_length + 1, step_size):
            end = start + segment_length
            ppg_segments.append(ppg_signal[start:end])
            resp_segments.append(resp_signal[start:end])
        
        return np.array(ppg_segments), np.array(resp_segments)
    
    def preprocess_subject_data(self, df: pd.DataFrame, 
                               subject_id: str) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess data for a single subject."""
        # Extract PPG and respiratory signals
        ppg_signal = df[self.data_config['input_column']].values
        resp_signal = df[self.data_config['target_column']].values
        
        print(f"Subject {subject_id} - Original data: PPG shape={ppg_signal.shape}, RESP shape={resp_signal.shape}")
        print(f"  PPG NaN count: {np.isnan(ppg_signal).sum()}, RESP NaN count: {np.isnan(resp_signal).sum()}")
        
        # Remove NaN values
        valid_indices = ~(np.isnan(ppg_signal) | np.isnan(resp_signal))
        ppg_signal = ppg_signal[valid_indices]
        resp_signal = resp_signal[valid_indices]
        
        if len(ppg_signal) == 0:
            print(f"Warning: No valid data for subject {subject_id}")
            return np.array([]), np.array([])
        
        print(f"  After NaN removal: PPG shape={ppg_signal.shape}, RESP shape={resp_signal.shape}")
        
        # Apply bandpass filter
        original_rate = self.data_config['sampling_rate']
        ppg_filtered = self.apply_bandpass_filter(ppg_signal, original_rate)
        resp_filtered = self.apply_bandpass_filter(resp_signal, original_rate)
        
        print(f"  After filtering: PPG NaN={np.isnan(ppg_filtered).sum()}, RESP NaN={np.isnan(resp_filtered).sum()}")
        
        # Downsample
        target_rate = self.preprocess_config['downsample']['target_rate']
        ppg_downsampled = self.downsample_signal(ppg_filtered, original_rate, target_rate)
        resp_downsampled = self.downsample_signal(resp_filtered, original_rate, target_rate)
        
        print(f"  After downsampling: PPG shape={ppg_downsampled.shape}, RESP shape={resp_downsampled.shape}")
        print(f"  PPG NaN={np.isnan(ppg_downsampled).sum()}, RESP NaN={np.isnan(resp_downsampled).sum()}")
        
        # Normalize
        norm_method = self.preprocess_config['normalization']
        ppg_normalized = self.normalize_signal(ppg_downsampled, norm_method)
        resp_normalized = self.normalize_signal(resp_downsampled, norm_method)
        
        print(f"  After normalization: PPG NaN={np.isnan(ppg_normalized).sum()}, RESP NaN={np.isnan(resp_normalized).sum()}")
        print(f"  PPG stats: min={ppg_normalized.min():.4f}, max={ppg_normalized.max():.4f}, mean={ppg_normalized.mean():.4f}")
        print(f"  RESP stats: min={resp_normalized.min():.4f}, max={resp_normalized.max():.4f}, mean={resp_normalized.mean():.4f}")
        
        # Segment
        segment_length = self.data_config['segment_length'] // (original_rate // target_rate)
        overlap = self.data_config['overlap']
        
        ppg_segments, resp_segments = self.segment_signal(
            ppg_normalized, resp_normalized, segment_length, overlap
        )
        
        print(f"Subject {subject_id}: {len(ppg_segments)} segments created")
        
        # Final check for NaN in segments
        if len(ppg_segments) > 0:
            ppg_nan_count = np.isnan(ppg_segments).sum()
            resp_nan_count = np.isnan(resp_segments).sum()
            print(f"  Final segments: PPG NaN={ppg_nan_count}, RESP NaN={resp_nan_count}")
        
        return ppg_segments, resp_segments
    
    def prepare_dataset(self, csv_folder: str) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Prepare the complete dataset for all subjects."""
        # Load all CSV files
        csv_files = self.load_csv_files(csv_folder)
        
        # Preprocess each subject
        processed_data = {}
        for subject_id, df in csv_files.items():
            ppg_segments, resp_segments = self.preprocess_subject_data(df, subject_id)
            if len(ppg_segments) > 0:
                processed_data[subject_id] = (ppg_segments, resp_segments)
        
        return processed_data


def create_subject_wise_splits(processed_data: Dict[str, Tuple[np.ndarray, np.ndarray]], 
                              test_subject: str = None, 
                              val_split: float = 0.2, 
                              random_seed: int = 42) -> Dict:
    """
    Create subject-wise train/validation/test splits.
    
    Args:
        processed_data: Dictionary mapping subject IDs to (PPG, respiratory) data
        test_subject: Specific subject to use as test set. If None, uses leave-one-out CV
        val_split: Fraction of training subjects to use for validation
        random_seed: Random seed for reproducible splits
        
    Returns:
        Dictionary containing split information and data
    """
    subjects = list(processed_data.keys())
    
    if test_subject is not None:
        if test_subject not in subjects:
            raise ValueError(f"Test subject '{test_subject}' not found in dataset. Available: {subjects}")
        
        # Single fold with specified test subject
        remaining_subjects = [s for s in subjects if s != test_subject]
        
        # Split remaining subjects into train and validation
        if len(remaining_subjects) < 2:
            raise ValueError(f"Not enough subjects for train/val split. Need at least 2, got {len(remaining_subjects)}")
        
        # Set random seed for reproducible splits
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        n_val_subjects = max(1, int(len(remaining_subjects) * val_split))
        val_subjects = random.sample(remaining_subjects, n_val_subjects)
        train_subjects = [s for s in remaining_subjects if s not in val_subjects]
        
        splits = [{
            'fold_id': 0,
            'train_subjects': train_subjects,
            'val_subjects': val_subjects,
            'test_subject': test_subject
        }]
        
    else:
        # Leave-one-out cross-validation with subject-wise validation splits
        splits = []
        
        for i, test_subject in enumerate(subjects):
            remaining_subjects = [s for s in subjects if s != test_subject]
            
            if len(remaining_subjects) < 2:
                print(f"Warning: Not enough subjects for proper train/val split in fold {i}")
                train_subjects = remaining_subjects
                val_subjects = []
            else:
                # Set random seed for reproducible splits
                random.seed(random_seed + i)  # Different seed for each fold
                np.random.seed(random_seed + i)
                
                n_val_subjects = max(1, int(len(remaining_subjects) * val_split))
                val_subjects = random.sample(remaining_subjects, n_val_subjects)
                train_subjects = [s for s in remaining_subjects if s not in val_subjects]
            
            splits.append({
                'fold_id': i,
                'train_subjects': train_subjects,
                'val_subjects': val_subjects,
                'test_subject': test_subject
            })
    
    return splits


def prepare_subject_wise_fold_data(processed_data: Dict[str, Tuple[np.ndarray, np.ndarray]], 
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


def print_split_summary(splits: List[Dict], processed_data: Dict[str, Tuple[np.ndarray, np.ndarray]]):
    """Print a summary of the data splits."""
    print(f"\n{'='*60}")
    print(f"SUBJECT-WISE DATA SPLIT SUMMARY")
    print(f"{'='*60}")
    print(f"Total subjects: {len(processed_data)}")
    print(f"Total folds: {len(splits)}")
    
    # Calculate total segments per subject
    subject_segments = {}
    for subject_id, (ppg_segments, _) in processed_data.items():
        subject_segments[subject_id] = len(ppg_segments)
    
    print(f"\nSegments per subject:")
    for subject_id, n_segments in sorted(subject_segments.items()):
        print(f"  {subject_id}: {n_segments} segments")
    
    print(f"\nSplit details:")
    for split in splits:
        fold_id = split['fold_id']
        train_subjects = split['train_subjects']
        val_subjects = split['val_subjects']
        test_subject = split['test_subject']
        
        train_segments = sum(subject_segments.get(s, 0) for s in train_subjects)
        val_segments = sum(subject_segments.get(s, 0) for s in val_subjects)
        test_segments = subject_segments.get(test_subject, 0)
        
        print(f"\n  Fold {fold_id}:")
        print(f"    Train subjects ({len(train_subjects)}): {train_subjects}")
        print(f"    Train segments: {train_segments}")
        print(f"    Val subjects ({len(val_subjects)}): {val_subjects}")
        print(f"    Val segments: {val_segments}")
        print(f"    Test subject: {test_subject}")
        print(f"    Test segments: {test_segments}")
    
    print(f"\n{'='*60}")


# Backward compatibility functions
def create_cross_validation_splits(processed_data: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> List[Dict]:
    """Create leave-one-out cross-validation splits (backward compatibility)."""
    splits = create_subject_wise_splits(processed_data, test_subject=None)
    
    # Convert to old format for backward compatibility
    old_format_splits = []
    for split in splits:
        # Combine train and val subjects for backward compatibility
        all_train_subjects = split['train_subjects'] + split['val_subjects']
        old_format_splits.append({
            'train_subjects': all_train_subjects,
            'test_subject': split['test_subject'],
            'fold_id': split['fold_id']
        })
    
    return old_format_splits


def prepare_fold_data(processed_data: Dict[str, Tuple[np.ndarray, np.ndarray]], 
                     train_subjects: List[str], test_subject: str,
                     val_split: float = 0.2) -> Dict:
    """Prepare data for a specific fold (backward compatibility with random validation split)."""
    # This maintains the old behavior for backward compatibility
    # but still ensures subject-wise splitting
    
    # Split train_subjects into actual train and validation subjects
    if len(train_subjects) < 2:
        print("Warning: Not enough training subjects for proper subject-wise validation split")
        actual_train_subjects = train_subjects
        val_subjects = []
    else:
        n_val_subjects = max(1, int(len(train_subjects) * val_split))
        val_subjects = random.sample(train_subjects, n_val_subjects)
        actual_train_subjects = [s for s in train_subjects if s not in val_subjects]
    
    return prepare_subject_wise_fold_data(
        processed_data, actual_train_subjects, val_subjects, test_subject
    )
