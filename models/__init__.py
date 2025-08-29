from .cnn1d import CNN1D, ResidualCNN1D
from .lstm import LSTM, BiLSTM
from .cnn_lstm import CNN_LSTM, AttentionCNN_LSTM
from .rwkv import RWKV, ImprovedTransformer, WaveNet
from .improved_rwkv import RevisedRWKV_DualBranch
from .dual_mode import DualModeWrapper, DualModeModel


def get_model(model_name: str, task_mode: str = "signal", **kwargs):
    """Factory function to get model by name with dual-mode support."""
    """Factory function to get model by name."""
    
    # Dual-mode specific models
    dual_mode_models = {
        'DualModeModel': DualModeModel,
    }
    
    # Base models that can be wrapped for dual-mode
    base_models = {
        'CNN1D': CNN1D,
        'ResidualCNN1D': ResidualCNN1D,
        'LSTM': LSTM,
        'BiLSTM': BiLSTM,
        'CNN_LSTM': CNN_LSTM,
        'AttentionCNN_LSTM': AttentionCNN_LSTM,
        'RWKV': RWKV,
        'ImprovedTransformer': ImprovedTransformer,
        'WaveNet': WaveNet,
        'RevisedRWKV_DualBranch': RevisedRWKV_DualBranch
    }
    
    # Check if it's a dual-mode specific model
    if model_name in dual_mode_models:
        return dual_mode_models[model_name](task_mode=task_mode, **kwargs)
    
    # Check if it's a base model
    elif model_name in base_models:
        
        if task_mode == "rate":
            # For rate mode, create base model first, then wrap it
            model_kwargs = {k: v for k, v in kwargs.items() if k != 'task_mode'}
            base_model = base_models[model_name](**model_kwargs)
            return DualModeWrapper(base_model, task_mode=task_mode, input_size=kwargs.get('input_size', 512))
        else:
            # Return base model as-is for signal estimation
            model_kwargs = {k: v for k, v in kwargs.items() if k != 'task_mode'}
            return base_models[model_name](**model_kwargs)
    
    else:
        all_models = list(dual_mode_models.keys()) + list(base_models.keys())
        raise ValueError(f"Model {model_name} not found. Available models: {all_models}")


__all__ = [
    'CNN1D', 'ResidualCNN1D', 'LSTM', 'BiLSTM', 
    'CNN_LSTM', 'AttentionCNN_LSTM', 'RWKV', 
    'ImprovedTransformer', 'WaveNet', 'RevisedRWKV_DualBranch',
    'DualModeWrapper', 'DualModeModel', 'get_model'
]
