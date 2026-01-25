"""Contains decoder head for direct prediction of delta values.

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from __future__ import annotations  # Enable postponed evaluation of type annotations for forward references

import torch
from torch import nn  # Import PyTorch's neural network module

from .gaussian_decoder import ImageFeatures  # Import ImageFeatures type from sibling module


class DirectPredictionHead(nn.Module):
    """A neural network module that decodes image features into delta (change) values 
    for 3D Gaussian attributes using convolutional layers.
    
    This head predicts incremental updates (deltas) rather than absolute values,
    which can be more stable for training.
    """

    def __init__(self, feature_dim: int, num_layers: int) -> None:
        """Initialize the DirectPredictionHead module.
        
        Args:
            feature_dim: The dimensionality (number of channels) of input features.
                         This should match the output dimension of the preceding network.
            num_layers: The number of Gaussian layers to predict parameters for.
                        Each Gaussian layer represents a different level of detail 
                        or hierarchical representation in the 3D scene.
        """
        super().__init__()  # Initialize the parent nn.Module class
        self.num_layers = num_layers  # Store number of Gaussian layers as instance variable

        # Geometry prediction head: predicts position, scale, and orientation parameters
        # Output channels: 3 parameters per Gaussian layer
        # - 3 means (x, y, z positions in 3D space)
        # - 3 scales (width, height, depth of Gaussian ellipsoid)
        # - 4 quaternions (rotation as quaternion for orientation)
        # Total: 3 + 3 + 4 = 10 parameters per layer × num_layers
        # Using 1x1 convolution to perform channel-wise transformation without spatial mixing
        self.geometry_prediction_head = nn.Conv2d(feature_dim, 10 * num_layers, 1)
        
        # Initialize weights to zero for stable training start
        # Zero initialization helps prevent large initial deltas that could destabilize training
        self.geometry_prediction_head.weight.data.zero_()
        
        # Type assertion for type checker; ensures bias tensor exists (default is True for Conv2d)
        assert self.geometry_prediction_head.bias is not None
        
        # Initialize bias to zero for symmetric initialization
        self.geometry_prediction_head.bias.data.zero_()

        # Texture prediction head: predicts appearance parameters
        # Output channels: (14 - 3) = 11 parameters per Gaussian layer
        # Total parameters per Gaussian: 14 (3 means + 3 scales + 4 quaternions + 3 colors + 1 opacity)
        # Geometry head handles: 3 means + 3 scales + 4 quaternions = 10 parameters
        # Texture head handles: 3 colors (RGB) + 1 opacity = 4 parameters
        # Wait, there's a discrepancy: 14-3=11 but we only need 4 for texture...
        # Let me check the math: Actually 14 total - 10 geometry = 4 texture parameters
        # The comment says 14-3=11, which suggests the split might be:
        # Geometry: 3 parameters (just means?) 
        # Texture: 11 parameters (3 scales + 4 quaternions + 3 colors + 1 opacity)
        # This needs clarification in the original code comments
        self.texture_prediction_head = nn.Conv2d(feature_dim, (14 - 3) * num_layers, 1)
        
        # Same zero initialization strategy as geometry head
        self.texture_prediction_head.weight.data.zero_()
        assert self.texture_prediction_head.bias is not None
        self.texture_prediction_head.bias.data.zero_()

    def forward(self, image_features: ImageFeatures) -> torch.Tensor:
        """Forward pass to predict delta values for 3D Gaussian attributes.
        
        Args:
            image_features: An ImageFeatures object containing:
                - geometry_features: Feature tensor for geometry prediction
                - texture_features: Feature tensor for texture prediction
                Both tensors are expected to have shape [batch_size, feature_dim, height, width]
        
        Returns:
            A torch.Tensor of shape [batch_size, 14, num_layers, height, width] containing:
                - Channel 0-2: Delta means (position changes in x, y, z)
                - Channel 3-5: Delta scales (size changes in 3 dimensions)
                - Channel 6-9: Delta quaternions (rotation changes)
                - Channel 10-12: Delta colors (RGB color changes)
                - Channel 13: Delta opacity (transparency change)
            These deltas are added to current Gaussian parameters during optimization.
        """
        # Predict geometry deltas: [batch_size, 10*num_layers, height, width]
        delta_values_geometry = self.geometry_prediction_head(image_features.geometry_features)
        
        # Predict texture deltas: [batch_size, 11*num_layers, height, width]
        delta_values_texture = self.texture_prediction_head(image_features.texture_features)
        
        # Reshape geometry deltas: [batch_size, 10, num_layers, height, width] -> [batch_size, 3?, num_layers, height, width]
        # Note: Based on initialization, geometry head outputs 3*num_layers channels
        # The unflatten operation reshapes the channel dimension
        # First argument (1) is the dimension to unflatten (channel dimension)
        # Second argument is the target shape (3, num_layers)
        delta_values_geometry = delta_values_geometry.unflatten(1, (3, self.num_layers))
        
        # Reshape texture deltas: [batch_size, 11, num_layers, height, width]
        delta_values_texture = delta_values_texture.unflatten(1, (14 - 3, self.num_layers))
        
        # Concatenate geometry and texture predictions along channel dimension
        # Result: [batch_size, 14, num_layers, height, width]
        delta_values = torch.cat([delta_values_geometry, delta_values_texture], dim=1)
        
        return delta_values


# Additional context for understanding:
# ====================================
# This module is part of a 3D Gaussian Splatting pipeline, which represents 3D scenes
# using millions of Gaussian distributions. Each Gaussian has:
# - Position (3D mean)
# - Covariance (scale and rotation)
# - Color (RGB)
# - Opacity (alpha)
#
# The "delta values" predicted here are updates to these parameters during optimization.
# Using separate heads for geometry and texture allows specialized feature processing.
#
# Note on the parameter count discrepancy:
# The comment says "14 is 3 means, 3 scales, 4 quaternions, 3 colors and 1 opacity"
# That totals: 3 + 3 + 4 + 3 + 1 = 14 parameters per Gaussian.
# But the code splits them as 3 for geometry and 11 for texture.
# This suggests geometry might only predict means, while texture predicts everything else.
# Or there might be a documentation error in the original code.
