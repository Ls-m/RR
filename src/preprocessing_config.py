import os
import json
import pickle
import numpy as np
from typing import Dict, Any, Optional
from datetime import datetime


class PreprocessingConfigManager:
    """Manager for saving and loading preprocessing configurations."""
    
    def __init__(self, config_dir: str = "preprocessing_configs"):
        self.config_dir = config_dir
        os.makedirs(config_dir, exist_ok=True)
    
    def save_preprocessing_config(self, config: Dict[str, Any], 
                                 preprocessor_stats: Optional[Dict] = None,
                                 config_name: str = None) -> str:
        """
        Save preprocessing configuration and statistics.
        
        Args:
            config: Full configuration dictionary
            preprocessor_stats: Statistics from preprocessing (means, stds, etc.)
            config_name: Optional name for the config file
            
        Returns:
            Path to saved configuration file
        """
        if config_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            config_name = f"preprocess_config_{timestamp}"
        
        # Extract relevant preprocessing settings
        preprocessing_config = {
            'data': config.get('data', {}),
            'preprocessing': config.get('preprocessing', {}),
            'timestamp': datetime.now().isoformat(),
            'config_name': config_name
        }
        
        # Add preprocessor statistics if provided
        if preprocessor_stats:
            preprocessing_config['stats'] = preprocessor_stats
        
        # Save as JSON for human readability
        json_path = os.path.join(self.config_dir, f"{config_name}.json")
        with open(json_path, 'w') as f:
            json.dump(preprocessing_config, f, indent=2, default=self._json_serializer)
        
        # Also save as pickle for exact reproduction
        pickle_path = os.path.join(self.config_dir, f"{config_name}.pkl")
        with open(pickle_path, 'wb') as f:
            pickle.dump(preprocessing_config, f)
        
        print(f"Preprocessing config saved to:")
        print(f"  JSON: {json_path}")
        print(f"  Pickle: {pickle_path}")
        
        return json_path
    
    def load_preprocessing_config(self, config_name: str, 
                                 format: str = 'json') -> Dict[str, Any]:
        """
        Load preprocessing configuration.
        
        Args:
            config_name: Name of the configuration (without extension)
            format: 'json' or 'pickle'
            
        Returns:
            Loaded preprocessing configuration
        """
        if format == 'json':
            config_path = os.path.join(self.config_dir, f"{config_name}.json")
            with open(config_path, 'r') as f:
                config = json.load(f)
        elif format == 'pickle':
            config_path = os.path.join(self.config_dir, f"{config_name}.pkl")
            with open(config_path, 'rb') as f:
                config = pickle.load(f)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        print(f"Loaded preprocessing config from: {config_path}")
        return config
    
    def list_saved_configs(self) -> list:
        """List all saved preprocessing configurations."""
        configs = []
        for filename in os.listdir(self.config_dir):
            if filename.endswith('.json'):
                config_name = filename[:-5]  # Remove .json extension
                configs.append(config_name)
        return sorted(configs)
    
    def merge_configs(self, base_config: Dict[str, Any], 
                     saved_config_name: str,
                     adjustable_params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Merge a saved preprocessing config with a base config, allowing adjustable parameters.
        
        Args:
            base_config: Base configuration dictionary
            saved_config_name: Name of saved preprocessing config
            adjustable_params: Parameters that can be adjusted (e.g., segment_length)
            
        Returns:
            Merged configuration
        """
        saved_config = self.load_preprocessing_config(saved_config_name)
        
        # Start with base config
        merged_config = base_config.copy()
        
        # Update with saved preprocessing settings
        merged_config['data'] = saved_config.get('data', {})
        merged_config['preprocessing'] = saved_config.get('preprocessing', {})
        
        # Apply adjustable parameters if provided
        if adjustable_params:
            for key, value in adjustable_params.items():
                if '.' in key:
                    # Handle nested keys like 'data.segment_length'
                    keys = key.split('.')
                    current = merged_config
                    for k in keys[:-1]:
                        if k not in current:
                            current[k] = {}
                        current = current[k]
                    current[keys[-1]] = value
                else:
                    # Handle top-level keys
                    if key in merged_config:
                        merged_config[key] = value
        
        print(f"Merged config with saved preprocessing settings from: {saved_config_name}")
        if adjustable_params:
            print(f"Applied adjustable parameters: {adjustable_params}")
        
        return merged_config
    
    def _json_serializer(self, obj):
        """Custom JSON serializer for numpy arrays and other objects."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, datetime):
            return obj.isoformat()
        else:
            return str(obj)


class EnhancedDataPreprocessor:
    """Enhanced data preprocessor with statistics tracking."""
    
    def __init__(self, config: Dict, config_manager: PreprocessingConfigManager = None):
        self.config = config
        self.data_config = config['data']
        self.preprocess_config = config['preprocessing']
        self.config_manager = config_manager or PreprocessingConfigManager()
        
        # Statistics tracking
        self.stats = {
            'subjects_processed': 0,
            'total_segments': 0,
            'preprocessing_stats': {},
            'signal_stats': {}
        }
    
    def compute_signal_statistics(self, signal: np.ndarray, signal_name: str):
        """Compute and store signal statistics."""
        if signal_name not in self.stats['signal_stats']:
            self.stats['signal_stats'][signal_name] = {
                'samples': [],
                'means': [],
                'stds': [],
                'mins': [],
                'maxs': []
            }
        
        stats = self.stats['signal_stats'][signal_name]
        stats['samples'].append(len(signal))
        stats['means'].append(np.mean(signal))
        stats['stds'].append(np.std(signal))
        stats['mins'].append(np.min(signal))
        stats['maxs'].append(np.max(signal))
    
    def get_aggregated_stats(self) -> Dict[str, Any]:
        """Get aggregated statistics across all processed signals."""
        aggregated = {
            'subjects_processed': self.stats['subjects_processed'],
            'total_segments': self.stats['total_segments'],
            'signal_stats': {}
        }
        
        for signal_name, stats in self.stats['signal_stats'].items():
            aggregated['signal_stats'][signal_name] = {
                'total_samples': sum(stats['samples']),
                'avg_length': np.mean(stats['samples']),
                'global_mean': np.mean(stats['means']),
                'global_std': np.mean(stats['stds']),
                'global_min': np.min(stats['mins']),
                'global_max': np.max(stats['maxs'])
            }
        
        return aggregated
    
    def save_preprocessing_setup(self, config_name: str = None) -> str:
        """Save the current preprocessing setup with statistics."""
        aggregated_stats = self.get_aggregated_stats()
        return self.config_manager.save_preprocessing_config(
            self.config, 
            aggregated_stats, 
            config_name
        )


def create_preprocessing_config_from_saved(saved_config_name: str,
                                         base_config_path: str = "configs/config.yaml",
                                         adjustable_params: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Create a new configuration using saved preprocessing settings.
    
    Args:
        saved_config_name: Name of saved preprocessing configuration
        base_config_path: Path to base configuration file
        adjustable_params: Parameters to adjust (e.g., {'data.segment_length': 1024})
        
    Returns:
        New configuration dictionary
    """
    import yaml
    
    # Load base config
    with open(base_config_path, 'r') as f:
        base_config = yaml.safe_load(f)
    
    # Create config manager and merge
    config_manager = PreprocessingConfigManager()
    merged_config = config_manager.merge_configs(
        base_config, 
        saved_config_name, 
        adjustable_params
    )
    
    return merged_config


# Example usage functions
def save_current_preprocessing_setup(config: Dict[str, Any], 
                                   preprocessor_stats: Dict = None,
                                   name: str = None) -> str:
    """Convenience function to save current preprocessing setup."""
    config_manager = PreprocessingConfigManager()
    return config_manager.save_preprocessing_config(config, preprocessor_stats, name)


def load_and_adjust_preprocessing_config(saved_name: str,
                                       segment_length: int = None,
                                       other_adjustments: Dict = None) -> Dict[str, Any]:
    """
    Convenience function to load and adjust preprocessing config.
    
    Args:
        saved_name: Name of saved configuration
        segment_length: New segment length if different
        other_adjustments: Other parameters to adjust
        
    Returns:
        Adjusted configuration
    """
    adjustable_params = {}
    
    if segment_length is not None:
        adjustable_params['data.segment_length'] = segment_length
    
    if other_adjustments:
        adjustable_params.update(other_adjustments)
    
    return create_preprocessing_config_from_saved(
        saved_name, 
        adjustable_params=adjustable_params if adjustable_params else None
    )
