"""Contains Sliding Pyramid Network architecture.

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.fx
import torch.nn as nn
import torch.nn.functional as F

from sharp.utils.training import checkpoint_wrapper

from .base_encoder import BaseEncoder
from .vit_encoder import TimmViT

# torch.fx.wrap is used here to mark functions as leaf nodes during symbolic tracing
# ensuring they are not traced but seen as atomic operation. In short, symbolic tracing
# struggles with native python functions and conditional flows.
non_traceable_ops = ("len", "int")
for op in non_traceable_ops:
    torch.fx.wrap(op)


class SlidingPyramidNetwork(BaseEncoder):
    """Sliding Pyramid Network.

    An encoder aimed at creating multi-resolution encodings from Vision Transformers.

    Reference: Bochkovskii et al. - "Depth pro: Sharp monocular metric depth in less
               than a second." (ICLR 2024)
    """

    def __init__(
        self,
        dims_encoder: Iterable[int],
        patch_encoder: TimmViT,
        image_encoder: TimmViT,
        use_patch_overlap: bool = True,
    ):
        """Initialize Sliding Pyramid Network.

        The framework
            1. creates an image pyramid,
            2. generates overlapping patches with a sliding window at each pyramid level,
            3. creates batched encodings via vision transformer backbones,
            4. produces multi-resolution encodings.

        Args:
            dims_encoder: Dimensions of the encoder at different layers.
            patch_encoder: Backbone used for highres part of the pyramid.
            image_encoder: Backbone used for lowres part of the pyramid.
            use_patch_overlap: Whether to use overlap between patches in SPN.
        """
        super().__init__()

        self.dim_in = patch_encoder.dim_in

        self.dims_encoder = list(dims_encoder)
        self.patch_encoder = patch_encoder
        self.image_encoder = image_encoder

        base_embed_dim = patch_encoder.embed_dim
        lowres_embed_dim = image_encoder.embed_dim
        self.patch_size = patch_encoder.internal_resolution()

        self.grad_checkpointing = False
        self.use_patch_overlap = use_patch_overlap

        # Retrieve intermediate feature ids registered in create_monodepth_encoder.
        self.patch_intermediate_features_ids = patch_encoder.intermediate_features_ids
        if (
            not isinstance(self.patch_intermediate_features_ids, list)
            or not len(self.patch_intermediate_features_ids) == 4
        ):
            raise ValueError("Patch intermediate feature ids must be a 4-item list.")

        self.image_intermediate_features_ids = image_encoder.intermediate_features_ids

        def _create_project_upsample_block(
            dim_in: int,
            dim_out: int,
            upsample_layers: int,
            dim_intermediate=None,
        ) -> nn.Module:
            if dim_intermediate is None:
                dim_intermediate = dim_out
            # Projection.
            blocks = [
                nn.Conv2d(
                    in_channels=dim_in,
                    out_channels=dim_intermediate,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                )
            ]

            # Upsampling.
            blocks += [
                nn.ConvTranspose2d(
                    in_channels=dim_intermediate if i == 0 else dim_out,
                    out_channels=dim_out,
                    kernel_size=2,
                    stride=2,
                    padding=0,
                    bias=False,
                )
                for i in range(upsample_layers)
            ]

            return nn.Sequential(*blocks)

        self.upsample_latent0 = _create_project_upsample_block(
            dim_in=base_embed_dim,
            dim_out=self.dims_encoder[0],
            upsample_layers=3,
            dim_intermediate=self.dims_encoder[1],
        )
        self.upsample_latent1 = _create_project_upsample_block(
            dim_in=base_embed_dim, dim_out=self.dims_encoder[1], upsample_layers=2
        )

        self.upsample0 = _create_project_upsample_block(
            dim_in=base_embed_dim, dim_out=self.dims_encoder[2], upsample_layers=1
        )
        self.upsample1 = _create_project_upsample_block(
            dim_in=base_embed_dim, dim_out=self.dims_encoder[3], upsample_layers=1
        )
        self.upsample2 = _create_project_upsample_block(
            dim_in=base_embed_dim, dim_out=self.dims_encoder[4], upsample_layers=1
        )

        self.upsample_lowres = nn.ConvTranspose2d(
            in_channels=lowres_embed_dim,
            out_channels=self.dims_encoder[4],
            kernel_size=2,
            stride=2,
            padding=0,
            bias=True,
        )
        self.fuse_lowres = nn.Conv2d(
            in_channels=(self.dims_encoder[4] + self.dims_encoder[4]),
            out_channels=self.dims_encoder[4],
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

    def internal_resolution(self) -> int:
        """Return the full image size of the SPN network."""
        return self.patch_size * 4

    @torch.jit.ignore
    def set_grad_checkpointing(self, is_enabled=True):
        """Enable grad checkpointing."""
        self.grad_checkpointing = is_enabled
        self.patch_encoder.set_grad_checkpointing(is_enabled)
        self.image_encoder.set_grad_checkpointing(is_enabled)

    @torch.jit.ignore
    def set_requires_grad_(self, patch_encoder: bool, image_encoder: bool):
        """Set requires grad for separate components."""
        self.patch_encoder.requires_grad_(patch_encoder)
        self.image_encoder.requires_grad_(image_encoder)

        # Always freeze the unused TimmViT head to exclude it from the calculation of
        # trainable parameters.
        self.patch_encoder.head.requires_grad_(False)
        self.image_encoder.head.requires_grad_(False)

        # These upsamplers only affect patch encoder's feature maps.
        self.upsample_latent0.requires_grad_(patch_encoder)
        self.upsample_latent1.requires_grad_(patch_encoder)
        self.upsample0.requires_grad_(patch_encoder)
        self.upsample1.requires_grad_(patch_encoder)
        self.upsample2.requires_grad_(patch_encoder)

        # This upsampler affects only image encoder's feature map.
        self.upsample_lowres.requires_grad_(image_encoder)

        # This fuser affects both image and patch encoders.
        self.fuse_lowres.requires_grad_(image_encoder or patch_encoder)

    def _create_pyramid(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Creates a 3-level image pyramid."""
        # Original resolution: 1536 by default.
        x0 = x

        # Middle resolution: 768 by default.
        x1 = F.interpolate(x, size=None, scale_factor=0.5, mode="bilinear", align_corners=False)

        # Low resolution: 384 by default, corresponding to the backbone resolution.
        x2 = F.interpolate(x, size=None, scale_factor=0.25, mode="bilinear", align_corners=False)

        return x0, x1, x2

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Encode input at multiple resolutions."""
        batch_size = x.shape[0]

        # Step 0: create a 3-level image pyramid.
        x0, x1, x2 = self._create_pyramid(x)

        if self.use_patch_overlap:
            # Step 1: split to create batched overlapped mini-images at the ViT
            # resolution.
            # 5x5 @ 384x384 at the highest resolution (1536x1536).
            x0_patches = split(x0, overlap_ratio=0.25, patch_size=self.patch_size)
            # 3x3 @ 384x384 at the middle resolution (768x768).
            x1_patches = split(x1, overlap_ratio=0.5, patch_size=self.patch_size)
            # 1x1 # 384x384 at the lowest resolution (384x384).
            x2_patches = x2
            padding = 3
        else:
            # Step 1: split to create batched overlapped mini-images at the ViT
            # resolution.
            # 4x4 @ 384x384 at the highest resolution (1536x1536).
            x0_patches = split(x0, overlap_ratio=0.0, patch_size=self.patch_size)
            # 2x2 @ 384x384 at the middle resolution (768x768).
            x1_patches = split(x1, overlap_ratio=0.0, patch_size=self.patch_size)
            # 1x1 # 384x384 at the lowest resolution (384x384).
            x2_patches = x2
            padding = 0
        x0_tile_size = x0_patches.shape[0]

        # Concatenate all the sliding window patches and form a batch of size
        # (35=5x5+3x3+1x1) or (21=4x4+2x2+1x1).
        x_pyramid_patches = torch.cat(
            (x0_patches, x1_patches, x2_patches),
            dim=0,
        )

        # Run the ViT model and get the result of large batch size.
        #
        # For the retrieval of intermediate features forward hooks are more concise,
        # but they are not well compatible with symbolic tracing because attributes
        # of submodules can be lost during tracing. Therefore, forward hooks may not
        # be preserved during graph transformation, leading to unexpected behavior.
        # To avoid such issues it is safer not to use them because they are not
        # essential here.
        x_pyramid_encodings, patch_intermediate_features = self.patch_encoder(x_pyramid_patches)

        # Step 3: merging.
        # Merge highres latent encoding.
        # NOTE: list type check has completed in init.
        x_latent0_encodings = self.patch_encoder.reshape_feature(
            patch_intermediate_features[self.patch_intermediate_features_ids[0]]  # type:ignore[index]
        )
        x_latent0_features = merge(
            x_latent0_encodings[: x0_tile_size],
            batch_size=batch_size,
            padding=padding,
        )

        x_latent1_encodings = self.patch_encoder.reshape_feature(
            patch_intermediate_features[self.patch_intermediate_features_ids[1]]  # type:ignore[index]
        )
        x_latent1_features = merge(
            x_latent1_encodings[: x0_tile_size],
            batch_size=batch_size,
            padding=padding,
        )

        # Split the 35 batch size from pyramid encoding back into 5x5+3x3+1x1.
        x0_encodings, x1_encodings, x2_encodings = torch.split(
            x_pyramid_encodings,
            [len(x0_patches), len(x1_patches), len(x2_patches)],
            dim=0,
        )

        # 96x96 feature maps by merging 5x5 @ 24x24 patches with overlaps.
        x0_features = merge(x0_encodings, batch_size=batch_size, padding=padding)

        # 48x84 feature maps by merging 3x3 @ 24x24 patches with overlaps.
        x1_features = merge(x1_encodings, batch_size=batch_size, padding=2 * padding)

        # 24x24 feature maps.
        x2_features = x2_encodings

        # Apply the image encoder.
        x_lowres_features, image_intermediate_features = self.image_encoder(x2_patches)

        # Upsample feature maps.
        x_latent0_features = checkpoint_wrapper(self, self.upsample_latent0, x_latent0_features)
        x_latent1_features = checkpoint_wrapper(self, self.upsample_latent1, x_latent1_features)

        x0_features = checkpoint_wrapper(self, self.upsample0, x0_features)
        x1_features = checkpoint_wrapper(self, self.upsample1, x1_features)
        x2_features = checkpoint_wrapper(self, self.upsample2, x2_features)

        x_lowres_features = checkpoint_wrapper(self, self.upsample_lowres, x_lowres_features)
        x_lowres_features = checkpoint_wrapper(
            self, self.fuse_lowres, torch.cat((x2_features, x_lowres_features), dim=1)
        )

        output = [
            x_latent0_features,
            x_latent1_features,
            x0_features,
            x1_features,
            x_lowres_features,
        ]

        return output


# It seems that torch.fx.wrap can only be applied to functions, not methods.
# Hence, split and merge were converted into functions to be marked as atomic
# operations for symbolic tracing.
@torch.fx.wrap
def split(image: torch.Tensor, overlap_ratio: float = 0.25, patch_size: int = 384) -> torch.Tensor:
    """Split the input into small patches with sliding window."""
    patch_stride = int(patch_size * (1 - overlap_ratio))

    image_size = image.shape[-1]
    steps = int(math.ceil((image_size - patch_size) / patch_stride)) + 1

    x_patch_list = []
    for j in range(steps):
        j0 = j * patch_stride
        j1 = j0 + patch_size

        for i in range(steps):
            i0 = i * patch_stride
            i1 = i0 + patch_size
            x_patch_list.append(image[..., j0:j1, i0:i1])

    return torch.cat(x_patch_list, dim=0)


# Decorator marking function as an atomic operator for symbolic tracing.
@torch.fx.wrap
def merge(image_patches: torch.Tensor, batch_size: int, padding: int = 3) -> torch.Tensor:
    """Merge the patched input into a image with sliding window."""
    steps = int(math.sqrt(image_patches.shape[0] // batch_size))

    idx = 0

    output_list = []
    for j in range(steps):
        output_row_list = []
        for i in range(steps):
            output = image_patches[batch_size * idx : batch_size * (idx + 1)]

            if padding != 0:
                if j != 0:
                    output = output[..., padding:, :]
                if i != 0:
                    output = output[..., :, padding:]
                if j != steps - 1:
                    output = output[..., :-padding, :]
                if i != steps - 1:
                    output = output[..., :, :-padding]

            output_row_list.append(output)
            idx += 1

        output_row = torch.cat(output_row_list, dim=-1)
        output_list.append(output_row)
    output = torch.cat(output_list, dim=-2)
    return output
