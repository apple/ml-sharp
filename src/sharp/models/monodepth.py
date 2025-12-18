"""Contains Dense Transformer Prediction architecture.

Implements a variant of Vision Transformers for Dense Prediction, https://arxiv.org/abs/2103.13413

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from __future__ import annotations

import copy
from typing import NamedTuple, Tuple

import torch
import torch.nn as nn

from sharp.models import normalizers
from sharp.models.decoders import MultiresConvDecoder, create_monodepth_decoder
from sharp.models.encoders import (
    SlidingPyramidNetwork,
    create_monodepth_encoder,
)
from sharp.utils import module_surgery

from .params import MonodepthAdaptorParams, MonodepthParams

DimsDecoder = Tuple[int, int, int, int, int]


class MonodepthDensePredictionTransformer(nn.Module):
    """Dense Prediction Transformer for monodepth.

    Attach the disparity prediction head for monodepth prediction.
    """

    def __init__(
        self,
        encoder: SlidingPyramidNetwork,
        decoder: MultiresConvDecoder,
        last_dims: tuple[int, int],
    ):
        """Initialize Dense Prediction Transformer.

        Args:
            encoder: The SlidingPyramidTransformer backbone.
            decoder: The MultiresConvDecoder decoder.
            last_dims: The dimension for the last convolution layers.
        """
        super().__init__()

        self.normalizer = normalizers.AffineRangeNormalizer(
            input_range=(0, 1), output_range=(-1, 1)
        )
        self.encoder = encoder
        self.decoder = decoder

        dim_decoder = decoder.dim_out
        self.head = nn.Sequential(
            nn.Conv2d(dim_decoder, dim_decoder // 2, kernel_size=3, stride=1, padding=1),
            nn.ConvTranspose2d(
                in_channels=dim_decoder // 2,
                out_channels=dim_decoder // 2,
                kernel_size=2,
                stride=2,
                padding=0,
                bias=True,
            ),
            nn.Conv2d(
                dim_decoder // 2,
                last_dims[0],
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(True),
            nn.Conv2d(last_dims[0], last_dims[1], kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
        )

        # Set the final convolution layer's bias to be 0.
        self.head[4].bias.data.fill_(0)

        self.grad_checkpointing = False

    @torch.jit.ignore
    def set_grad_checkpointing(self, is_enabled=True):
        """Enable grad checkpointing."""
        self.grad_checkpointing = is_enabled
        self.encoder.set_grad_checkpointing(self.grad_checkpointing)
        self.decoder.set_grad_checkpointing(self.grad_checkpointing)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Decode by projection and fusion of multi-resolution encodings."""
        encodings = self.encoder(self.normalizer(image))
        num_encoder_features = len(self.encoder.dims_encoder)
        features = self.decoder(encodings[:num_encoder_features])
        disparity = self.head(features)
        return disparity

    def internal_resolution(self) -> int:
        """Return the internal image size of the network."""
        return self.encoder.internal_resolution()


def create_monodepth_dpt(
    params: MonodepthParams | None = None,
) -> MonodepthDensePredictionTransformer:
    """Creates DepthDensePredictionTransformer model.

    Args:
        params: Parameters of monodepth network.

    Returns:
        The configured monodepth DPT.
    """
    if params is None:
        params = MonodepthParams()
    encoder: SlidingPyramidNetwork = create_monodepth_encoder(
        params.patch_encoder_preset,
        params.image_encoder_preset,
        use_patch_overlap=params.use_patch_overlap,
        last_encoder=params.dims_decoder[0],
    )

    decoder: MultiresConvDecoder = create_monodepth_decoder(
        params.patch_encoder_preset, params.dims_decoder
    )

    monodepth_model = MonodepthDensePredictionTransformer(
        encoder=encoder, decoder=decoder, last_dims=(32, 1)
    )

    # By default, we don't train the monodepth model.
    # However, we allow to selectively unfreeze parts of the network.
    monodepth_model.requires_grad_(False)

    monodepth_model.encoder.set_requires_grad_(
        patch_encoder=params.unfreeze_patch_encoder,
        image_encoder=params.unfreeze_image_encoder,
    )
    monodepth_model.decoder.requires_grad_(params.unfreeze_decoder)
    monodepth_model.head.requires_grad_(params.unfreeze_head)

    if not params.unfreeze_norm_layers:
        module_surgery.freeze_norm_layer(monodepth_model)

    monodepth_model.set_grad_checkpointing(params.grad_checkpointing)

    return monodepth_model


class MonodepthOutput(NamedTuple):
    """Output of the monodepth model."""

    # Disparity output from the monodepth model.
    disparity: torch.Tensor
    # Multi-level features from monodepth encoder.
    encoder_features: list[torch.Tensor]
    # Single-level feature from monodepth decoder.
    decoder_features: torch.Tensor
    # List of monodepth features to be used in gaussian predictor.
    output_features: list[torch.Tensor]
    # List of intermediate encoder features to be used in distillation.
    intermediate_features: list[torch.Tensor] = []


class MonodepthWithEncodingAdaptor(nn.Module):
    """Monodepth model with feature maps."""

    def __init__(
        self,
        monodepth_predictor: MonodepthDensePredictionTransformer,
        return_encoder_features: bool,
        return_decoder_features: bool,
        num_monodepth_layers: int,
        sorting_monodepth: bool,
    ):
        """Initialize MonodepthWithEncodingAdaptor.

        Args:
            monodepth_predictor: The monodepth model.
            return_encoder_features: Whether to return encoder features from monodepth model.
            return_decoder_features: Whether to return decoder features from monodepth model.
            num_monodepth_layers: How many layers the monodepth model predicts.
            sorting_monodepth: Whether to sort the monodepth output (for two layer monodepth).
        """
        super().__init__()
        self.monodepth_predictor = monodepth_predictor
        self.return_encoder_features = return_encoder_features
        self.return_decoder_features = return_decoder_features
        self.num_monodepth_layers = num_monodepth_layers
        self.sorting_monodepth = sorting_monodepth

    def forward(self, image: torch.Tensor) -> MonodepthOutput:
        """Process image and return disparity and feature maps."""
        inputs = self.monodepth_predictor.normalizer(image)
        encoder_output = self.monodepth_predictor.encoder(inputs)

        num_encoder_features = len(self.monodepth_predictor.encoder.dims_encoder)

        # NOTE: whether intermediate features are empty have already been decided
        # in monodepth_predictor during create_monodepth_dpt.
        encoder_features = encoder_output[:num_encoder_features]
        intermediate_features = encoder_output[num_encoder_features:]
        decoder_features = self.monodepth_predictor.decoder(encoder_features)
        disparity = self.monodepth_predictor.head(decoder_features)

        # We cannot use disparity.shape[1], otherwise the tracer will fail.
        if self.num_monodepth_layers == 2 and self.sorting_monodepth:
            first_layer_disparity = disparity.max(dim=1, keepdims=True).values
            second_layer_disparity = disparity.min(dim=1, keepdims=True).values
            disparity = torch.cat([first_layer_disparity, second_layer_disparity], dim=1)

        output_features = []
        if self.return_encoder_features:
            output_features.extend(encoder_features)

        if self.return_decoder_features:
            output_features.append(decoder_features)

        return MonodepthOutput(
            disparity=disparity,
            encoder_features=encoder_features,
            decoder_features=decoder_features,
            output_features=output_features,
            intermediate_features=intermediate_features,
        )

    def get_feature_dims(self) -> list[int]:
        """Return dimensions of output feature maps."""
        dims = []
        if self.return_encoder_features:
            dims.extend(self.monodepth_predictor.encoder.dims_encoder)

        if self.return_decoder_features:
            dims.append(self.monodepth_predictor.decoder.dim_out)

        return dims

    def internal_resolution(self) -> int:
        """Return the internal image size of the network."""
        return self.monodepth_predictor.internal_resolution()

    def replicate_head(self, num_repeat: int):
        """Replicate the last convolution layer (head[4] in DPT) for multi layer depth."""
        conv_last = copy.deepcopy(self.monodepth_predictor.head[4])
        self.monodepth_predictor.head[4].out_channels = num_repeat
        self.monodepth_predictor.head[4].weight = nn.Parameter(
            conv_last.weight.repeat(num_repeat, 1, 1, 1)
        )
        self.monodepth_predictor.head[4].bias = nn.Parameter(conv_last.bias.repeat(num_repeat))


def create_monodepth_adaptor(
    monodepth_predictor: MonodepthDensePredictionTransformer,
    params: MonodepthAdaptorParams,
    num_monodepth_layers: int,
    sorting_monodepth: bool,
) -> MonodepthWithEncodingAdaptor:
    """Create an adaptor that returns both disparity and features."""
    adaptor = MonodepthWithEncodingAdaptor(
        monodepth_predictor=monodepth_predictor,
        return_encoder_features=params.encoder_features,
        return_decoder_features=params.decoder_features,
        num_monodepth_layers=num_monodepth_layers,
        sorting_monodepth=sorting_monodepth,
    )
    return adaptor
