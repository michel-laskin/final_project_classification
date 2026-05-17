"""
Search Space Definitions for Optuna Hyperparameter Optimization

Defines the default parameter ranges and provides utilities
to sample hyperparameters from an Optuna trial.
"""

from typing import Dict, Any, List


# Default search space configuration.
# Each entry: (type, default_value, range_or_choices, enabled)
# Types: 'int', 'float', 'log_float', 'categorical'
DEFAULT_SEARCH_SPACE = {
    # --- Windowing ---
    'hrv_window_size': {
        'type': 'categorical',
        'default': 200,
        'choices': [100, 150, 200, 250, 300, 350, 400],
        'enabled': True,
        'group': 'Windowing',
        'description': 'Number of RR intervals per window',
    },
    'hrv_overlap_ratio': {
        'type': 'float',
        'default': 0.9,
        'low': 0.5,
        'high': 0.95,
        'step': 0.05,
        'enabled': True,
        'group': 'Windowing',
        'description': 'Overlap ratio between consecutive windows',
    },
    
    # --- Feature Encoder Architecture ---
    'embedding_dim': {
        'type': 'categorical',
        'default': 32,
        'choices': [16, 32, 48, 64],
        'enabled': True,
        'group': 'Encoder',
        'description': 'Output dimension of the feature encoder',
    },
    'encoder_num_layers': {
        'type': 'int',
        'default': 1,
        'low': 0,
        'high': 2,
        'enabled': True,
        'group': 'Encoder',
        'description': 'Number of hidden layers in encoder (0=direct projection)',
    },
    'encoder_hidden_dim': {
        'type': 'categorical',
        'default': 64,
        'choices': [32, 64, 96, 128],
        'enabled': True,
        'group': 'Encoder',
        'description': 'Width of hidden layers in encoder',
    },
    'encoder_activation': {
        'type': 'categorical',
        'default': 'silu',
        'choices': ['relu', 'silu', 'gelu'],
        'enabled': True,
        'group': 'Encoder',
        'description': 'Activation function in encoder',
    },
    
    # --- TCN Architecture ---
    'tcn_num_blocks': {
        'type': 'int',
        'default': 3,
        'low': 1,
        'high': 4,
        'enabled': True,
        'group': 'TCN',
        'description': 'Number of temporal blocks in TCN',
    },
    'tcn_channel_base': {
        'type': 'categorical',
        'default': 32,
        'choices': [16, 32, 48, 64],
        'enabled': True,
        'group': 'TCN',
        'description': 'Base channel width for TCN blocks',
    },
    'tcn_channel_growth': {
        'type': 'categorical',
        'default': 'double',
        'choices': ['constant', 'double'],
        'enabled': True,
        'group': 'TCN',
        'description': 'How channels grow across TCN blocks',
    },
    'tcn_kernel_size': {
        'type': 'int',
        'default': 3,
        'low': 2,
        'high': 7,
        'enabled': True,
        'group': 'TCN',
        'description': 'Kernel size for temporal convolutions',
    },
    
    # --- Training ---
    'learning_rate': {
        'type': 'log_float',
        'default': 0.0003,
        'low': 1e-5,
        'high': 1e-2,
        'enabled': True,
        'group': 'Training',
        'description': 'Adam optimizer learning rate',
    },
    'weight_decay': {
        'type': 'log_float',
        'default': 0.05,
        'low': 1e-4,
        'high': 0.1,
        'enabled': True,
        'group': 'Training',
        'description': 'L2 regularization strength',
    },
    'dropout': {
        'type': 'float',
        'default': 0.5,
        'low': 0.1,
        'high': 0.7,
        'step': 0.05,
        'enabled': True,
        'group': 'Training',
        'description': 'Dropout probability',
    },
    'batch_size': {
        'type': 'categorical',
        'default': 32,
        'choices': [16, 32, 64],
        'enabled': True,
        'group': 'Training',
        'description': 'Training batch size',
    },
    'group_lasso_lambda': {
        'type': 'log_float',
        'default': 0.01,
        'low': 1e-4,
        'high': 0.1,
        'enabled': True,
        'group': 'Training',
        'description': 'Group lasso regularization strength',
    },
}


def get_groups() -> List[str]:
    """Get ordered list of parameter groups."""
    seen = []
    for param in DEFAULT_SEARCH_SPACE.values():
        g = param['group']
        if g not in seen:
            seen.append(g)
    return seen


def sample_params(trial, search_space: Dict[str, Dict] = None) -> Dict[str, Any]:
    """
    Sample hyperparameters from an Optuna trial using the search space config.
    
    Disabled parameters use their default values.
    
    Args:
        trial: Optuna trial object.
        search_space: Search space configuration dict. Uses DEFAULT if None.
        
    Returns:
        Dictionary of sampled parameter values.
    """
    if search_space is None:
        search_space = DEFAULT_SEARCH_SPACE
    
    params = {}
    
    for name, spec in search_space.items():
        if not spec.get('enabled', True):
            # Use default value for disabled parameters
            params[name] = spec['default']
            continue
        
        param_type = spec['type']
        
        if param_type == 'int':
            params[name] = trial.suggest_int(
                name, spec['low'], spec['high']
            )
        elif param_type == 'float':
            step = spec.get('step', None)
            params[name] = trial.suggest_float(
                name, spec['low'], spec['high'], step=step
            )
        elif param_type == 'log_float':
            params[name] = trial.suggest_float(
                name, spec['low'], spec['high'], log=True
            )
        elif param_type == 'categorical':
            params[name] = trial.suggest_categorical(
                name, spec['choices']
            )
    
    return params


def build_tcn_channels(params: Dict[str, Any]) -> List[int]:
    """
    Build TCN channel list from sampled parameters.
    
    Args:
        params: Sampled parameters dict.
        
    Returns:
        List of channel sizes, e.g. [32, 64, 128].
    """
    base = params['tcn_channel_base']
    n_blocks = params['tcn_num_blocks']
    growth = params['tcn_channel_growth']
    
    if growth == 'constant':
        return [base] * n_blocks
    elif growth == 'double':
        return [base * (2 ** i) for i in range(n_blocks)]
    else:
        return [base] * n_blocks
