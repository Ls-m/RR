import os
import glob
import pandas as pd
import numpy as np
from scipy import signal
from scipy.stats import zscore
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from typing import Tuple, List, Dict
# from vmdpy import VMD
from itertools import groupby
from scipy.signal import find_peaks
from scipy.stats import skew
import warnings
warnings.filterwarnings('ignore')


class DataPreprocessor:
    """Data preprocessing utilities for PPG to respiratory waveform estimation."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.data_config = config['data']
        self.preprocess_config = config['preprocessing']

    def load_csv_files(self, csv_folder: str) -> Dict[str, pd.DataFrame]:
        """Load all CSV files from the specified folder."""
        csv_files = {}
        csv_path = csv_folder
        if csv_path == 'bidmc_data':
            csv_path += '/bidmc_csv'
        print("csv path is ",csv_path)
        for filename in os.listdir(csv_path):
            if filename.endswith('.csv'):
                subject_id = filename.replace('.csv', '')
                df = pd.read_csv(os.path.join(csv_path, filename))
                try:
                    # Try reading as tab-separated first, then comma-separated
                    # try:
                    #     df = pd.read_csv(os.path.join(csv_path, filename), sep='\t')
                        
                    # except:
                    #     df = pd.read_csv(os.path.join(csv_path, filename))
                    
                    # Skip the first row which contains sampling rates
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
    
    # def denoise_signal(self, signal_data: np.ndarray ) -> np.ndarray:
    #     if len(signal_data) == 0 or np.all(np.isnan(signal_data)):
    #         return signal_data
    #     alpha = 2000       # Moderate bandwidth constraint
    #     tau = 0.            # Noise-tolerance (Lagrange multiplier)
    #     K = 5              # Number of modes to decompose into
    #     DC = 0             # No DC part imposed
    #     init = 1           # Initialize omegas uniformly
    #     tol = 1e-7         # Convergence tolerance
    #     u, u_hat, omega = VMD(signal_data, alpha, tau, K, DC, init, tol)
    #     denoised_ppg = np.sum(u[:3, :], axis=0)
    #     return denoised_ppg
    def denoise_signals(self, ppg_signal: np.ndarray, resp_signal: np.ndarray, fs: int) -> Tuple[np.ndarray, np.ndarray]:
        """Denoise PPG signal using Envelope PPG Denoising Algorithm (EPDA) and apply same removals to resp signal."""
        if len(ppg_signal) == 0:
            return ppg_signal, resp_signal
        
        # Hardcoded parameters (can be moved to config)
        window_size_seconds = 30  # Process in 30-second windows
        motion_threshold = 2
        flat_height_threshold = 0.05  # Small amplitude for flat lines (adjusted for normalized signal)
        flat_temporal_threshold = 1.0  # Minimum duration for flat line in seconds
        gap_threshold_samples = int(0.5 * fs)  # Merge segments if gap < 0.5 seconds
        
        window_length = int(window_size_seconds * fs)
        ppg_denoised_parts = []
        resp_denoised_parts = []
        
        for start in range(0, len(ppg_signal), window_length):
            end = min(start + window_length, len(ppg_signal))
            ppg_window = ppg_signal[start:end]
            resp_window = resp_signal[start:end]
            
            if len(ppg_window) < fs:  # Skip small leftover
                ppg_denoised_parts.append(ppg_window)
                resp_denoised_parts.append(resp_window)
                continue
            
            # Step 1: Bandpass filter (0.5-10 Hz, 4th order)
            sos = signal.butter(4, [0.5 / (fs / 2), 10 / (fs / 2)], btype='band', output='sos')
            ppg_filt = signal.sosfiltfilt(sos, ppg_window)
            
            # Step 2: Calculate upper and lower envelopes
            distance = 0.2 * len(ppg_filt)
            peaks, _ = find_peaks(ppg_filt, distance=distance)
            troughs, _ = find_peaks(-ppg_filt, distance=distance)
            
            # Handle if no peaks/troughs
            if len(peaks) < 2 or len(troughs) < 2:
                ppg_denoised_parts.append(ppg_window)
                resp_denoised_parts.append(resp_window)
                continue
            
            upper_env = np.interp(np.arange(len(ppg_filt)), peaks, ppg_filt[peaks])
            lower_env = np.interp(np.arange(len(ppg_filt)), troughs, ppg_filt[troughs])
            
            env_diff = np.abs(upper_env - lower_env)
            
            # Calculate median
            median_env = np.median(env_diff)
            
            # Step 3: Calculate thresholds for motion
            q1, q3 = np.percentile(env_diff, [25, 75])
            iqr = q3 - q1
            upper_th = q3 + motion_threshold * iqr
            lower_th = max(0, q1 - motion_threshold * iqr)  # Clip lower to 0 since env_diff >=0
            
            # Step 4: Detect motion anomalies (slope change points out of thresholds)
            diffs = np.diff(env_diff)
            sign_diff = np.diff(np.sign(diffs))
            slope_changes = np.where(sign_diff != 0)[0] + 1
            
            motion_anomalies = [idx for idx in slope_changes if env_diff[idx] > upper_th or env_diff[idx] < lower_th]
            
            # Find crossings with median
            crossings = np.where(np.diff(np.sign(env_diff - median_env)))[0] + 1
            
            # For each motion anomaly, find surrounding crossings for segment
            motion_segments = []
            for anom_idx in motion_anomalies:
                left_cross = crossings[crossings < anom_idx]
                right_cross = crossings[crossings > anom_idx]
                if len(left_cross) > 0 and len(right_cross) > 0:
                    start_seg = max(left_cross[-1] - 1, 0)  # Adjust for crossing index
                    end_seg = min(right_cross[0], len(env_diff) - 1)
                    motion_segments.append((start_seg, end_seg))
            
            # Step 5: Detect flat line segments on filtered PPG
            height_detection = np.where(np.abs(ppg_filt) < flat_height_threshold, 0, 1)
            flat_segments = []
            for key, grp in groupby(enumerate(height_detection), lambda x: x[1]):
                if key == 0:
                    grp_list = list(grp)
                    s = grp_list[0][0]
                    e = grp_list[-1][0]
                    if (e - s + 1) >= flat_temporal_threshold * fs:
                        flat_segments.append((s, e))
            
            # Combine all segments
            all_segments = motion_segments + flat_segments
            
            if not all_segments:
                ppg_denoised_parts.append(ppg_window)
                resp_denoised_parts.append(resp_window)
                continue
            
            # Sort segments by start
            all_segments.sort(key=lambda x: x[0])
            
            # Merge overlapping or close segments
            merged = []
            current_start, current_end = all_segments[0]
            for next_start, next_end in all_segments[1:]:
                if current_end + gap_threshold_samples >= next_start:
                    current_end = max(current_end, next_end)
                else:
                    merged.append((current_start, current_end))
                    current_start, current_end = next_start, next_end
            merged.append((current_start, current_end))
            
            # Create mask to keep data not in merged segments
            keep_mask = np.ones(len(ppg_window), dtype=bool)
            for s, e in merged:
                keep_mask[s:e+1] = False
            
            ppg_clean = ppg_window[keep_mask]
            resp_clean = resp_window[keep_mask]
            
            # Final lowpass filter on cleaned PPG (2nd order, 8 Hz)
            if len(ppg_clean) > 0:
                sos_low = signal.butter(2, 8 / (fs / 2), btype='low', output='sos')
                ppg_clean = signal.sosfiltfilt(sos_low, ppg_clean)
            
            ppg_denoised_parts.append(ppg_clean)
            resp_denoised_parts.append(resp_clean)
        
        # Concatenate all parts
        ppg_denoised = np.concatenate(ppg_denoised_parts)
        resp_denoised = np.concatenate(resp_denoised_parts)
        
        print(f"After denoising: PPG shape={ppg_denoised.shape}, RESP shape={resp_denoised.shape}")
        
        return ppg_denoised, resp_denoised
    
    def compute_sqi_skewness(self, segment: np.ndarray) -> float:
        """Compute Skewness SQI for a PPG segment."""
        if len(segment) == 0:
            return -np.inf  # Invalid
        return skew(segment)

    def compute_sqi_matching(self, segment: np.ndarray, fs: int) -> float:
        """Compute Matching SQI (concordance between two peak detectors)."""
        if len(segment) == 0:
            return 0.0
        
        # Billauer's algorithm: local maxima with min distance (assuming HR < 180 bpm)
        min_distance = fs // 3  # ~0.33s min distance
        peaks_billauer, _ = find_peaks(segment, distance=min_distance)
        
        # Bing's algorithm approximation: first derivative with adaptive threshold
        deriv = np.diff(segment)
        thresh = np.mean(deriv[deriv > 0]) + 0.5 * np.std(deriv[deriv > 0])  # Positive slopes
        peaks_deriv, _ = find_peaks(deriv, height=thresh)
        peaks_bing = peaks_deriv + 1  # Shift by 1 due to diff
        
        if len(peaks_billauer) == 0 or len(peaks_bing) == 0:
            return 0.0
        
        # Matching formula: intersection over min of the two sets
        intersection = len(set(peaks_billauer) & set(peaks_bing))
        m_sqi = intersection / min(len(peaks_billauer), len(peaks_bing))
        return m_sqi

    def compute_sqi_std_deriv(self, segment: np.ndarray) -> float:
        """Compute inverse std of second derivative (lower value indicates more noise)."""
        if len(segment) < 3:
            return np.inf  # Invalid, high "noise"
        second_deriv = np.diff(np.diff(segment))
        return np.std(second_deriv)

    # def segment_signal(self, ppg_signal: np.ndarray, resp_signal: np.ndarray,
    #                    segment_length: int, overlap: float, fs: int,
    #                    sqi_threshold_skew: float = 0.1,
    #                    sqi_threshold_match: float = 0.6,
    #                    sqi_threshold_std: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
    #     """Segment signals into overlapping windows and discard low-quality based on SQI."""
    #     step_size = int(segment_length * (1 - overlap))
        
    #     ppg_segments = []
    #     resp_segments = []
        
    #     for start in range(0, len(ppg_signal) - segment_length + 1, step_size):
    #         end = start + segment_length
    #         ppg_seg = ppg_signal[start:end]
    #         resp_seg = resp_signal[start:end]
            
    #         # Compute SQIs
    #         sqi_skew = self.compute_sqi_skewness(ppg_seg)
    #         sqi_match = self.compute_sqi_matching(ppg_seg, fs)
    #         sqi_std = self.compute_sqi_std_deriv(ppg_seg)
            
    #         # Discard if any SQI fails (tune thresholds as needed)
    #         if (sqi_skew < sqi_threshold_skew or
                
    #             sqi_std > sqi_threshold_std):  # High std means noisy
    #             print(f"Skew={sqi_skew:.2f}, Match={sqi_match:.2f}, Std={sqi_std:.2f}")

    #             continue
            
    #         ppg_segments.append(ppg_seg)
    #         resp_segments.append(resp_seg)
        
    #     return np.array(ppg_segments), np.array(resp_segments)
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
        print("hereeee, ", df.columns)
        if ' PLETH' in df.columns:
            ppg_signal = df[' PLETH'].values
            resp_signal = df[' RESP'].values
        else:
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
        original_rate = self.data_config['sampling_rate']
        # Denoise (using EPDA on PPG, apply removals to both)
        ppg_signal, resp_signal = self.denoise_signals(ppg_signal, resp_signal, original_rate)
        print(f"  After denoise: PPG NaN={np.isnan(ppg_signal).sum()}, RESP NaN={np.isnan(resp_signal).sum()}")
        print(f"  PPG stats: min={ppg_signal.min():.4f}, max={ppg_signal.max():.4f}, mean={ppg_signal.mean():.4f}")
        # Apply bandpass filter
        
        ppg_filtered = self.apply_bandpass_filter(ppg_signal, original_rate)
        resp_filtered = self.apply_bandpass_filter(resp_signal, original_rate)
        
        print(f"  After filtering: PPG NaN={np.isnan(ppg_filtered).sum()}, RESP NaN={np.isnan(resp_filtered).sum()}")
        
        # Downsample
        target_rate = self.preprocess_config['downsample']['target_rate']
        # ppg_downsampled = self.downsample_signal(ppg_filtered, original_rate, target_rate)
        # resp_downsampled = self.downsample_signal(resp_filtered, original_rate, target_rate)
        
        # print(f"  After downsampling: PPG shape={ppg_downsampled.shape}, RESP shape={resp_downsampled.shape}")
        # print(f"  PPG NaN={np.isnan(ppg_downsampled).sum()}, RESP NaN={np.isnan(resp_downsampled).sum()}")
        ppg_downsampled = ppg_filtered
        resp_downsampled = resp_filtered
        # Normalize
        norm_method = self.preprocess_config['normalization']
        ppg_normalized = self.normalize_signal(ppg_downsampled, norm_method)
        resp_normalized = self.normalize_signal(resp_downsampled, norm_method)
        
        print(f"  After normalization: PPG NaN={np.isnan(ppg_normalized).sum()}, RESP NaN={np.isnan(resp_normalized).sum()}")
        print(f"  PPG stats: min={ppg_normalized.min():.4f}, max={ppg_normalized.max():.4f}, mean={ppg_normalized.mean():.4f}")
        print(f"  RESP stats: min={resp_normalized.min():.4f}, max={resp_normalized.max():.4f}, mean={resp_normalized.mean():.4f}")

        # # Denoise
        # ppg_normalized = self.denoise_signal(ppg_normalized)
        # print(f"  After denoise: PPG NaN={np.isnan(ppg_normalized).sum()}, RESP NaN={np.isnan(resp_normalized).sum()}")
        # print(f"  PPG stats: min={ppg_normalized.min():.4f}, max={ppg_normalized.max():.4f}, mean={ppg_normalized.mean():.4f}")

        # Segment
        segment_length = self.data_config['segment_length'] // (original_rate // target_rate)
        overlap = self.data_config['overlap']
        
        # ppg_segments, resp_segments = self.segment_signal(
        #     ppg_normalized, resp_normalized, segment_length, overlap, target_rate
        # )
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
    
    def prepare_dual_mode_dataset(self, csv_folder: str, task_mode: str = "signal") -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Prepare dataset for both signal estimation (Mode 1) and rate estimation (Mode 2)."""
        
        if task_mode == "signal":
            # Mode 1: PPG -> Respiratory Signal (existing functionality)
            return self.prepare_dataset(csv_folder)
        
        elif task_mode == "rate":
            # Mode 2: PPG -> Respiratory Rate
            return self.prepare_rate_dataset(csv_folder)
        
        else:
            raise ValueError(f"Invalid task_mode: {task_mode}. Must be 'signal' or 'rate'")
    
    def prepare_rate_dataset(self, csv_folder: str) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Prepare dataset for respiratory rate estimation (Mode 2)."""
        # Load signal files for PPG input
        csv_files = self.load_csv_files(csv_folder)
        # print(csv_files['bidmc_49_Signals'])
        # Load RR files for rate targets
        if 'bidmc' in csv_folder.lower():
            rr_files = self.load_bidmc_rr_data(csv_folder)
        else:
            # For other datasets, assume RR is already in the main CSV files
            rr_files = csv_files
        
        processed_data = {}
        my_keys = ["_".join(subject_id.split("_")[:-1]) for subject_id in csv_files.keys()]
        for subject_id in my_keys:
            # result = "_".join(subject_id.split("_")[:-1])
            # subject_id = result
            print("result is",subject_id)
            if subject_id not in rr_files:
                print(f"Warning: No RR data found for subject {subject_id}")
                continue
            
            # Process PPG signal (same as Mode 1)
            signal_df = csv_files[subject_id+"_Signals"]
            ppg_segments, _ = self.preprocess_subject_data(signal_df, subject_id)
            
            if len(ppg_segments) == 0:
                continue
            # print("len signal_df is ",len(signal_df))
            # print("len ppg_segments is ",len(ppg_segments))
            # exit()
            # Process RR targets
            rr_targets = self.prepare_rr_targets(rr_files[subject_id], ppg_segments, subject_id)
            
            if len(rr_targets) == len(ppg_segments):
                
                processed_data[subject_id] = (ppg_segments, rr_targets)
            else:
                print(f"Warning: Mismatch in segment count for {subject_id}: PPG={len(ppg_segments)}, RR={len(rr_targets)}")
        
        return processed_data
    
    def prepare_rr_targets(self, rr_df: pd.DataFrame, ppg_segments: np.ndarray, subject_id: str) -> np.ndarray:
        """Prepare respiratory rate targets aligned with PPG segments."""
        
        # Calculate segment timing information
        original_rate = self.data_config['sampling_rate']
        target_rate = self.preprocess_config['downsample']['target_rate']
        segment_length_samples = ppg_segments.shape[-1]  # After downsampling
        segment_duration = segment_length_samples / target_rate
        print("segment duration is ", segment_duration)
        
        overlap = self.data_config['overlap']
        step_duration = segment_duration * (1 - overlap)
        
        # Create timestamps for each segment
        num_segments = len(ppg_segments)
        print("num_segments is ", num_segments)
        
        segment_timestamps = np.array([i * step_duration for i in range(num_segments)])
        
        # Align RR data with segments
        if 'bidmc' in self.data_config.get('csv_folder', '').lower():
            # BIDMC format: use Numerics.csv data
            print("go to align_rr")
            aligned_rr = self.align_rr_with_segments(rr_df, segment_timestamps, segment_duration)
        else:
            # Other datasets: assume RR is constant or extract from signal
            rr_column = self.data_config.get('target_rr_column', 'RESP')
            if rr_column in rr_df.columns:
                # Use provided RR values (constant assumption)
                mean_rr = rr_df[rr_column].mean()
                aligned_rr = np.full(num_segments, mean_rr)
            else:
                print(f"Warning: RR column '{rr_column}' not found for {subject_id}")
                aligned_rr = np.full(num_segments, np.nan)
        
        # Convert to proper format (single values per segment)
        rr_targets = aligned_rr.reshape(-1, 1)  # Shape: (num_segments, 1)
        # print("shape rr_targets is ", rr_targets.shape)
        # print("shape ppg segments is ", ppg_segments.shape)
        print(f"Subject {subject_id}: Prepared {len(rr_targets)} RR targets, mean={np.nanmean(rr_targets):.2f}")
        
        return rr_targets

    def load_bidmc_rr_data(self, csv_folder: str) -> Dict[str, pd.DataFrame]:
        """Load BIDMC respiratory rate data from Numerics.csv files."""
        csv_path = os.path.join(os.getcwd(), csv_folder, 'bidmc_csv')
        
        if not os.path.exists(csv_path):
            print(f"CSV path does not exist: {csv_path}")
            return {}
        
        rr_files = {}
        pattern = os.path.join(csv_path, '*_Numerics.csv')
        
        for filepath in glob.glob(pattern):
            filename = os.path.basename(filepath)
            subject_id = filename.replace('_Numerics.csv', '')
            
            try:
                df = pd.read_csv(filepath)
                
                # Convert to numeric, handling any non-numeric values
                for col in df.columns:
                    if col != 'Time [s]':  # Keep time column as is
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                
                rr_files[subject_id] = df
                print(f"Loaded RR data for {subject_id}: {df.shape}, Columns: {df.columns.tolist()}")
                
            except Exception as e:
                print(f"Error loading {filename}: {e}")
                
        return rr_files

    def align_rr_with_segments(self, rr_data: pd.DataFrame, segment_timestamps: np.ndarray, 
                              segment_duration: float) -> np.ndarray:
        """Align respiratory rate data with signal segments."""
        rr_column = self.data_config.get('bidmc_rr_column', ' RESP')
        print(rr_data.columns)
        if rr_column not in rr_data.columns:
            print(f"Warning: Column '{rr_column}' not found in RR data. Available: {rr_data.columns.tolist()}")
            return np.full(len(segment_timestamps), np.nan)
        
        norm_method = self.preprocess_config['normalization']
        # print("heyyyyyyyyyyyyyyyyyyyy")
        # print("Before Normalization: ", rr_data[rr_column])
        # rr_data[rr_column] = self.normalize_signal(rr_data[rr_column], norm_method)
        # print("Normalized RR data:", rr_data[rr_column])
        if np.isnan(rr_data[rr_column]).sum()>0:
            print("Warning:  values are NaN after normalization.")
        
        aligned_rr = []
        
        for start_time in segment_timestamps:
            
            end_time = start_time + segment_duration
            
            # Find RR values within this time window
            mask = (rr_data['Time [s]'] >= start_time) & (rr_data['Time [s]'] < end_time)
            segment_rr_data = rr_data.loc[mask, rr_column]
            
            if len(segment_rr_data) == 0:
                
                # No RR data in this segment - interpolate from nearest values
                nearest_idx = np.argmin(np.abs(rr_data['Time [s]'] - (start_time + end_time) / 2))
                rr_value = rr_data.iloc[nearest_idx][rr_column]
            else:
                # Use mean RR value for this segment
                rr_value = segment_rr_data.mean()
            
            aligned_rr.append(rr_value)
        
        return np.array(aligned_rr)


def create_cross_validation_splits(processed_data: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> List[Dict]:
    """Create leave-one-out cross-validation splits."""
    subjects = list(processed_data.keys())
    cv_splits = []
    
    for test_subject in subjects:
        train_subjects = [s for s in subjects if s != test_subject]
        
        cv_splits.append({
            'train_subjects': train_subjects,
            'test_subject': test_subject,
            'fold_id': len(cv_splits)
        })
    
    return cv_splits


def prepare_fold_data(processed_data: Dict[str, Tuple[np.ndarray, np.ndarray]], 
                     train_subjects: List[str], test_subject: str,
                     val_split: float = 0.2) -> Dict:
    """Prepare data for a specific fold."""
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
    
    # Test data
    test_ppg, test_resp = processed_data[test_subject]
    
    return {
        'train_ppg': train_ppg[train_indices],
        'train_resp': train_resp[train_indices],
        'val_ppg': train_ppg[val_indices],
        'val_resp': train_resp[val_indices],
        'test_ppg': test_ppg,
        'test_resp': test_resp,
        'test_subject': test_subject
    }
