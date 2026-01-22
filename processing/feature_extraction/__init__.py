"""
Feature Extraction Module

Provides modular feature extraction by domain.
"""

# Import modular feature extraction components
from .time_domain_features import TimeDomainFeatures
from .frequency_domain_features import FrequencyDomainFeatures
from .non_linear_features import NonLinearFeatures
from .feature_pipeline import (
    extract_all_features_from_rr,
    features_dict_to_tensor,
    get_feature_names,
    get_feature_group_indices
)

__all__ = [
    'TimeDomainFeatures',
    'FrequencyDomainFeatures',
    'NonLinearFeatures',
    'extract_all_features_from_rr',
    'features_dict_to_tensor',
    'get_feature_names',
    'get_feature_group_indices'
]
