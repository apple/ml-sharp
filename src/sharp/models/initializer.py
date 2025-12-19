"""Contains modules to initialize Gaussians from RGBD.

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn

from .params import ColorInitOption, DepthInitOption, InitializerParams


def create_initializer(params: InitializerParams) -> nn.Module:
    """Create inpainter."""
    return MultiLayerInitializer(
        num_layers=params.num_layers,
        stride=params.stride,
        base_depth=params.base_depth,
        scale_factor=params.scale_factor,
        disparity_factor=params.disparity_factor,
        color_option=params.color_option,
        first_layer_depth_option=params.first_layer_depth_option,
        rest_layer_depth_option=params.rest_layer_depth_option,
        normalize_depth=params.normalize_depth,
        feature_input_stop_grad=params.feature_input_stop_grad,
    )


class GaussianBaseValues(NamedTuple):
    """Base values for gaussian predictor.

    We predict x and y in normalized device coordinates (NDC) where (-1, -1) is the top
    left corner and (1, 1) the bottom right corner. The last component of
    mean_vectors_ndc is inverse depth.
    """

    mean_x_ndc: torch.Tensor
    mean_y_ndc: torch.Tensor
    mean_inverse_z_ndc: torch.Tensor

    scales: torch.Tensor
    quaternions: torch.Tensor
    colors: torch.Tensor
    opacities: torch.Tensor


class InitializerOutput(NamedTuple):
    """Output of initializer."""

    # Gaussian base values.
    gaussian_base_values: GaussianBaseValues

    # Feature input to the Gaussian predictor.
    feature_input: torch.Tensor

    # Global scale to unscale output.
    global_scale: torch.Tensor | None = None


class MultiLayerInitializer(nn.Module):
    """Initialize Gaussians with multilayer representation.

    The returned tensors have the shape

        batch_size x dim x num_layers x height x width

    where dim indicates the dimensionality of the property.
    Some of the dimensions might be set to 1 for efficiency reasons.
    """

    def __init__(
        self,
        num_layers: int,
        stride: int,
        base_depth: float,
        scale_factor: float,
        disparity_factor: float,
        color_option: ColorInitOption = "first_layer",
        first_layer_depth_option: DepthInitOption = "surface_min",
        rest_layer_depth_option: DepthInitOption = "surface_min",
        normalize_depth: bool = True,
        feature_input_stop_grad: bool = True,
    ) -> None:
        """Initialize MultilayerInitializer.

        Args:
            stride: The downsample rate of output feature map.
            base_depth: The depth of the first layer (after the foreground
                layer if use_depth=True).
            scale_factor: Multiply scale of Gaussians by this factor.
            disparity_factor: Factor to convert inverse depth to disparity.
            num_layers: How many layers of Gaussians to predict.
            color_option: Which color option to initialize the multi-layer gaussians.
            first_layer_depth_option: Which depth option to initialize the first layer of gaussians.
            rest_layer_depth_option: Which depth option to initialize the rest layers of gaussians.
            normalize_depth: # Whether to normalize depth to [DepthTransformParam.depth_min,
                DepthTransformParam.depth_max).
            feature_input_stop_grad: Whether to not propagate gradients through feature inputs.
        """
        super().__init__()
        self.num_layers = num_layers
        self.stride = stride
        self.base_depth = base_depth
        self.scale_factor = scale_factor
        self.disparity_factor = disparity_factor
        self.color_option = color_option
        self.first_layer_depth_option = first_layer_depth_option
        self.rest_layer_depth_option = rest_layer_depth_option
        self.normalize_depth = normalize_depth
        self.feature_input_stop_grad = feature_input_stop_grad

    def prepare_feature_input(self, image: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        """Prepare the feature input to the Gaussian predictor."""
        if self.feature_input_stop_grad:
            image = image.detach()
            depth = depth.detach()

        normalized_disparity = self.disparity_factor / depth
        features_in = torch.cat([image, normalized_disparity], dim=1)
        features_in = 2.0 * features_in - 1.0
        return features_in

    def forward(self, image: torch.Tensor, depth: torch.Tensor) -> InitializerOutput:
        """Construct Gaussian base values and prepare feature input.

        Args:
            image: The image to process.
            depth: The corresponding depth map from the monodepth network.

        Returns:
            The base value for Gaussians.
        """
        image = image.contiguous()
        depth = depth.contiguous()
        device = depth.device
        batch_size, _, image_height, image_width = depth.shape
        base_height, base_width = (
            image_height // self.stride,
            image_width // self.stride,
        )
        # global_scale is the inverse of the depth_factor, which is used to rescale
        # the depth such that it is numerically stable for training.
        global_scale: torch.Tensor | None = None
        if self.normalize_depth:
            depth, depth_factor = _rescale_depth(depth)
            global_scale = 1.0 / depth_factor

        def _create_disparity_layers(num_layers: int = 1) -> torch.Tensor:
            """Create multiple disparity layers."""
            disparity = torch.linspace(1.0 / self.base_depth, 0.0, num_layers + 1, device=device)
            return disparity[None, None, :-1, None, None].repeat(
                batch_size, 1, 1, base_height, base_width
            )

        def _create_surface_layer(
            depth: torch.Tensor,
            depth_pooling_mode: str,
        ) -> torch.Tensor:
            """Create multiple surface layers."""
            disparity = 1.0 / depth
            if depth_pooling_mode == "min":
                disparity = torch.max_pool2d(disparity, self.stride, self.stride)
            elif depth_pooling_mode == "max":
                disparity = -torch.max_pool2d(-disparity, self.stride, self.stride)
            else:
                raise ValueError(f"Invalid depth pooling mode {depth_pooling_mode}.")

            return disparity[:, :, None, :, :]

        # Input disparity dimensions:
        #   (batch_size, num_channels in (1, 2), height, width)

        # Output disparity dimensions:
        #   (batch_size, num_channels=1, num_layers in (1, 2), height, width)
        if self.first_layer_depth_option == "surface_min":
            first_disparity = _create_surface_layer(depth[:, 0:1], "min")
        elif self.first_layer_depth_option == "surface_max":
            first_disparity = _create_surface_layer(depth[:, 0:1], "max")
        elif self.first_layer_depth_option in ("base_depth", "linear_disparity"):
            first_disparity = _create_disparity_layers()
        else:
            raise ValueError(f"Unknown depth init option: {self.first_layer_depth_option}.")

        if self.num_layers == 1:
            disparity = first_disparity
        else:  # Fill in the rest layers.
            following_depth = depth if depth.shape[1] == 1 else depth[:, 1:]
            if self.rest_layer_depth_option == "surface_min":
                following_disparity = _create_surface_layer(following_depth, "min")
            elif self.rest_layer_depth_option == "surface_max":
                following_disparity = _create_surface_layer(following_depth, "max")
            elif self.rest_layer_depth_option == "base_depth":
                following_disparity = torch.cat(
                    [_create_disparity_layers() for i in range(self.num_layers - 1)],
                    dim=2,
                )
            elif self.rest_layer_depth_option == "linear_disparity":
                following_disparity = _create_disparity_layers(self.num_layers - 1)
            else:
                raise ValueError(f"Unknown depth init option: {self.rest_layer_depth_option}.")

            disparity = torch.cat([first_disparity, following_disparity], dim=2)

        # Prepare base values.
        base_x_ndc, base_y_ndc = _create_base_xy(depth, self.stride, self.num_layers)
        disparity_scale_factor = 2 * self.scale_factor * self.stride / float(image_width)
        base_scales = _create_base_scale(disparity, disparity_scale_factor)

        base_quaternions = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
        base_quaternions = base_quaternions[None, :, None, None, None]

        # Initializing the opacitiy this way ensures that the initial transmittance
        # is approximately
        #
        #       1 / e ~= (1 - 1 / self.num_layers)**self.num_layers
        #
        # and hence independent of the number of layers.
        #
        base_opacities = torch.tensor([min(1.0 / self.num_layers, 0.5)], device=device)
        base_colors = torch.empty(
            batch_size, 3, self.num_layers, base_height, base_width, device=device
        ).fill_(0.5)
        # Dimensions: (batch_size, num_channels, num_layers, height, width)
        if self.color_option == "none":
            pass
        elif self.color_option == "first_layer":
            base_colors[:, :, 0] = torch.nn.functional.avg_pool2d(image, self.stride, self.stride)
        elif self.color_option == "all_layers":
            temp = torch.nn.functional.avg_pool2d(image, self.stride, self.stride)
            base_colors = temp[:, :, None, :, :].repeat(1, 1, self.num_layers, 1, 1)
        else:
            raise ValueError(f"Unknown color init option: {self.color_option}.")

        features_in = self.prepare_feature_input(image, depth)
        base_gaussians = GaussianBaseValues(
            mean_x_ndc=base_x_ndc,
            mean_y_ndc=base_y_ndc,
            mean_inverse_z_ndc=disparity,
            scales=base_scales,
            quaternions=base_quaternions,
            colors=base_colors,
            opacities=base_opacities,
        )

        return InitializerOutput(
            gaussian_base_values=base_gaussians,
            feature_input=features_in,
            global_scale=global_scale,
        )


def _create_base_xy(
    depth: torch.Tensor, stride: int, num_layers: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create base x and y coordinates for the gaussians in NDC space."""
    device = depth.device
    batch_size, _, image_height, image_width = depth.shape
    xx = torch.arange(0.5 * stride, image_width, stride, device=device)
    yy = torch.arange(0.5 * stride, image_height, stride, device=device)
    xx = 2 * xx / image_width - 1.0
    yy = 2 * yy / image_height - 1.0

    xx, yy = torch.meshgrid(xx, yy, indexing="xy")
    base_x_ndc = xx[None, None, None].repeat(batch_size, 1, num_layers, 1, 1)
    base_y_ndc = yy[None, None, None].repeat(batch_size, 1, num_layers, 1, 1)

    return base_x_ndc, base_y_ndc


def _create_base_scale(disparity: torch.Tensor, disparity_scale_factor: float) -> torch.Tensor:
    """Create base scale for the gaussians."""
    inverse_disparity = torch.ones_like(disparity) / disparity
    base_scales = inverse_disparity * disparity_scale_factor
    return base_scales


def _rescale_depth(
    depth: torch.Tensor, depth_min: float = 1.0, depth_max: float = 1e2
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rescale a depth image tensor.

    Args:
        depth: The depth tensor to transform.
        depth_min: The min depth to scale depth to.
        depth_max: The max clamp depth after scaling.

    Returns:
        The rescaled depth and rescale factor.
    """
    current_depth_min = depth.flatten(depth.ndim - 3).min(dim=-1).values
    depth_factor = depth_min / (current_depth_min + 1e-6)
    depth = (depth * depth_factor[..., None, None, None]).clamp(max=depth_max)
    return depth, depth_factor
