"""
Group Lasso Regularization Module

Implements group lasso penalty for feature selection at the group level.
Encourages entire feature groups to be selected or excluded together.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Union


class GroupLassoRegularizer:
    """
    Group Lasso regularizer for neural network weights.
    
    Applies group lasso penalty to encourage sparsity at the group level:
    penalty = λ * Σ_g √|G_g| * ‖W_g‖₂
    
    Where:
    - W_g are weights corresponding to feature group g
    - |G_g| is the size of group g (used to balance group sizes)
    - ‖·‖₂ is the L2 norm
    """
    
    def __init__(
        self,
        group_indices: Dict[str, List[int]],
        lambda_reg: float = 0.01
    ):
        """
        Initialize the group lasso regularizer.
        
        Args:
            group_indices: Dict mapping group names to list of feature indices.
                          Example: {'time_domain': [0, 1, 5, 10], 'frequency_domain': [2, 3, 4]}
            lambda_reg: Regularization strength (default: 0.01).
        """
        self.group_indices = group_indices
        self.lambda_reg = lambda_reg
        
        # Precompute group sizes and their square roots for efficiency
        self.group_weights = {}
        for name, indices in group_indices.items():
            group_size = len(indices)
            # Square root of group size for balanced penalty
            self.group_weights[name] = np.sqrt(group_size) if group_size > 0 else 0
    
    def compute_penalty(self, weight_matrix: torch.Tensor) -> torch.Tensor:
        """
        Compute group lasso penalty for a weight matrix.
        
        Args:
            weight_matrix: Weight tensor of shape [out_features, in_features].
                          in_features should match the total number of features
                          across all groups.
        
        Returns:
            Scalar tensor representing the group lasso penalty.
        """
        penalty = torch.tensor(0.0, device=weight_matrix.device, dtype=weight_matrix.dtype)
        
        for name, indices in self.group_indices.items():
            if len(indices) == 0:
                continue
                
            # Extract weights for this group (all output neurons, selected input features)
            # indices is a list of feature column indices
            indices_tensor = torch.tensor(indices, device=weight_matrix.device, dtype=torch.long)
            group_weights = weight_matrix[:, indices_tensor]
            
            # Compute L2 norm of the group weights (Frobenius norm)
            group_norm = torch.norm(group_weights, p='fro')
            
            # Add weighted penalty (sqrt of group size for balance)
            penalty = penalty + self.group_weights[name] * group_norm
        
        return self.lambda_reg * penalty
    
    def penalty_from_model(
        self,
        model: nn.Module,
        encoder_name: str = 'comprehensive_features'
    ) -> torch.Tensor:
        """
        Compute group lasso penalty from a FusionModel's encoder weights.
        
        Targets the first linear layer of the specified encoder, which maps
        raw features to embedding space.
        
        Args:
            model: FusionModel instance.
            encoder_name: Name of the encoder in model.encoders dict.
        
        Returns:
            Scalar tensor representing the group lasso penalty.
        """
        # Access the encoder's first linear layer weight
        # FeatureEncoder structure: net.0 = Linear, net.1 = LayerNorm, etc.
        encoder = model.encoders[encoder_name]
        first_linear = encoder.net[0]  # nn.Linear
        weight_matrix = first_linear.weight  # [out_features, in_features]
        
        return self.compute_penalty(weight_matrix)
    
    def get_group_norms(
        self,
        model: nn.Module,
        encoder_name: str = 'comprehensive_features'
    ) -> Dict[str, float]:
        """
        Get the L2 norm of weights for each feature group.
        Useful for monitoring which groups are being regularized.
        
        Args:
            model: FusionModel instance.
            encoder_name: Name of the encoder in model.encoders dict.
        
        Returns:
            Dict mapping group names to their weight L2 norms.
        """
        encoder = model.encoders[encoder_name]
        first_linear = encoder.net[0]
        weight_matrix = first_linear.weight
        
        norms = {}
        for name, indices in self.group_indices.items():
            if len(indices) == 0:
                norms[name] = 0.0
                continue
            indices_tensor = torch.tensor(indices, device=weight_matrix.device, dtype=torch.long)
            group_weights = weight_matrix[:, indices_tensor]
            norms[name] = torch.norm(group_weights, p='fro').item()
        
        return norms
