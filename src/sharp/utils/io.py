"""Contains image IO.

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import IO, Any, Protocol

import imageio.v2 as iio
import numpy as np
import pillow_heif
import torch
from PIL import ExifTags, Image, TiffTags

from .vis import METRIC_DEPTH_MAX_CLAMP_METER, colorize_depth

LOGGER = logging.getLogger(__name__)


# NOTE: unused, kept for reference.
Image.MAX_IMAGE_PIXELS = 200000000


def load_rgb(
    path: Path, auto_rotate: bool = True, remove_alpha: bool = True
) -> tuple[np.ndarray, list[bytes] | None, float]:
    """Load an RGB image."""
    LOGGER.debug(f"Loading image {path} ...")

    if path.suffix.lower() in [".heic"]:
        heif_file = pillow_heif.open_heif(path, convert_hdr_to_8bit=True)
        img_pil = heif_file.to_pillow()
    else:
        img_pil = Image.open(path)

    img_exif = extract_exif(img_pil)
    icc_profile = img_pil.info.get("icc_profile", None)

    # Rotate the image.
    if auto_rotate:
        exif_orientation = img_exif.get("Orientation", 1)
        if exif_orientation == 3:
            img_pil = img_pil.transpose(Image.ROTATE_180)
        elif exif_orientation == 6:
            img_pil = img_pil.transpose(Image.ROTATE_270)
        elif exif_orientation == 8:
            img_pil = img_pil.transpose(Image.ROTATE_90)
        elif exif_orientation != 1:
            LOGGER.warning(f"Ignoring image orientation {exif_orientation}.")

    # Extract the focal length.
    f_35mm = img_exif.get("FocalLengthIn35mmFilm", img_exif.get("FocalLenIn35mmFilm", None))
    if f_35mm is None or f_35mm < 1:
        f_35mm = img_exif.get("FocalLength", None)
        if f_35mm is None:
            LOGGER.warning(f"Did not find focallength in exif data of {path} - Setting to 30mm.")
            f_35mm = 30.0
        if f_35mm < 10.0:
            LOGGER.info("Found focal length below 10mm, assuming it's not for 35mm.")
            # This is a very crude approximation.
            f_35mm *= 8.4

    img = np.asarray(img_pil)
    # Convert to RGB if single channel.
    if img.ndim < 3 or img.shape[2] == 1:
        img = np.dstack((img, img, img))

    if remove_alpha:
        img = img[:, :, :3]

    LOGGER.debug(f"\tHxW: {img.shape[0]}x{img.shape[1]}")
    LOGGER.debug(f"\tfocal length @ 35mm film: {f_35mm}mm")
    f_px = convert_focallength(img.shape[1], img.shape[0], f_35mm)
    LOGGER.debug(f"\tfocal length: {f_px:.2f}px")

    return img, icc_profile, f_px


def extract_exif(img_pil: Image.Image) -> dict[str, Any]:
    """Return exif information as a dictionary."""
    # Get full exif description from get_ifd(0x8769):
    # cf https://pillow.readthedocs.io/en/stable/releasenotes/8.2.0.html#image-getexif-exif-and-gps-ifd # noqa
    img_exif = img_pil.getexif().get_ifd(0x8769)
    exif_dict = {ExifTags.TAGS[k]: v for k, v in img_exif.items() if k in ExifTags.TAGS}

    # https://pillow.readthedocs.io/en/stable/_modules/PIL/TiffTags.html# # noqa
    tiff_tags = img_pil.getexif()
    tiff_dict = {TiffTags.TAGS_V2[k].name: v for k, v in tiff_tags.items() if k in TiffTags.TAGS_V2}
    return {**exif_dict, **tiff_dict}


def convert_focallength(width: float, height: float, f_mm: float = 30) -> float:
    """Converts a focal length given in mm to pixels."""
    return f_mm * np.sqrt(width**2.0 + height**2.0) / np.sqrt(36**2 + 24**2)


def save_image(
    image: np.ndarray,
    output_path: Path,
    icc_profile: list[bytes] | None = None,
    jpeg_quality: int = 92,
) -> None:
    """Save image to given path."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    extensions_to_format = Image.registered_extensions()
    try:
        format = extensions_to_format[output_path.suffix.lower()]
    except KeyError:
        raise ValueError(f"Unsupported output format {output_path.suffix}.")

    with output_path.open("wb") as file_handle:
        write_image(
            image,
            file_handle,
            format,
            icc_profile=icc_profile,
            jpeg_quality=jpeg_quality,
        )


def write_image(
    image: np.ndarray,
    output_io: IO[bytes],
    format="jpg",
    icc_profile: list[bytes] | None = None,
    jpeg_quality: int = 92,
):
    """Write image to binary stream."""
    pil_config = {}
    if format == "JPEG":
        pil_config["quality"] = jpeg_quality

    image_pil = Image.fromarray(image)

    # Workaround to error [io.UnsupportedOperation: seek].
    if format == "TIFF":
        bytes_io = io.BytesIO()
        image_pil.save(bytes_io, format="TIFF")
        bytes_io.seek(0)
        output_io.write(bytes_io.read())
        return

    image_pil.save(output_io, format, icc_profile=icc_profile, **pil_config)


def get_supported_image_extensions(with_heic: bool = True) -> list[str]:
    """Return supported image extensions."""
    exts = Image.registered_extensions()
    supported_extensions = {ex for ex, f in exts.items() if f in Image.OPEN}
    if with_heic:
        supported_extensions.add(".heic")

    supported_extensions_upper = {ex.upper() for ex in supported_extensions}
    return list(supported_extensions | supported_extensions_upper)


def get_supported_video_extensions():
    """Return supported video extensions."""
    supported_extensions = {".mp4", ".mov"}
    supported_extensions_upper = {ext.upper() for ext in supported_extensions}
    return list(supported_extensions | supported_extensions_upper)


class OutputWriter(Protocol):
    """Protocol for writing output to disk."""

    def add_frame(self, image: torch.Tensor, depth: torch.Tensor) -> None:
        """Add a single frame to output."""
        ...

    def close(self) -> None:
        """Finish writing."""
        ...


class VideoWriter(OutputWriter):
    """Output writer for video output."""

    def __init__(self, output_path: Path, fps: float = 30.0, render_depth: bool = True) -> None:
        """Initialize VideoWriter."""
        output_path.parent.mkdir(exist_ok=True, parents=True)
        self.output_path = output_path
        self.image_writer = iio.get_writer(output_path, fps=fps)

        self.max_depth_estimate = None
        if render_depth:
            self.depth_writer = iio.get_writer(output_path.with_suffix(".depth.mp4"), fps=fps)

    def add_frame(self, image: torch.Tensor, depth: torch.Tensor) -> None:
        """Add a single frame to output."""
        image_np = image.detach().cpu().numpy()
        self.image_writer.append_data(image_np)

        if self.depth_writer is not None:
            if self.max_depth_estimate is None:
                self.max_depth_estimate = depth.max().item()

            colored_depth_pt = colorize_depth(
                depth,
                min(self.max_depth_estimate, METRIC_DEPTH_MAX_CLAMP_METER),  # type: ignore[call-overload]
            )
            colored_depth_np = colored_depth_pt.squeeze(0).permute(1, 2, 0).cpu().numpy()
            self.depth_writer.append_data(colored_depth_np)

    def close(self):
        """Finish writing."""
        self.image_writer.close()
