"""Station transform utilities for multi-station 360 capture.

Converts station pose data (position + orientation quaternion) from
stations.json into 3x4 affine transforms compatible with
sharp.utils.gaussians.apply_transform().
"""

import json
import numpy as np
import torch
from pathlib import Path
from typing import Any


def quaternion_to_rotation_matrix(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Convert a unit quaternion (qx, qy, qz, qw) to a 3x3 rotation matrix.

    Uses the Hamilton convention where qw is the scalar part.

    Args:
        qx, qy, qz: Vector (imaginary) components.
        qw: Scalar (real) component.

    Returns:
        3x3 numpy rotation matrix.
    """
    # Normalize the quaternion
    norm = np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm < 1e-10:
        return np.eye(3)
    qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm

    # Rotation matrix from quaternion
    R = np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ], dtype=np.float64)

    return R


def build_station_transform(
    position_3d: dict[str, float],
    orientation_3d: dict[str, float],
    device: torch.device,
) -> torch.Tensor:
    """Build a 3x4 affine transform from station pose data.

    The transform maps points from the station's local coordinate frame
    (where the 360 camera is at the origin) into the global/world frame
    defined by the stations.json coordinate system.

    Args:
        position_3d: {"x": float, "y": float, "z": float}
        orientation_3d: {"qx": float, "qy": float, "qz": float, "qw": float}
        device: Torch device for the output tensor.

    Returns:
        3x4 torch.Tensor representing [R | t] (world_from_station).
    """
    R = quaternion_to_rotation_matrix(
        orientation_3d["qx"],
        orientation_3d["qy"],
        orientation_3d["qz"],
        orientation_3d["qw"],
    )

    transform = np.zeros((3, 4), dtype=np.float64)
    transform[:3, :3] = R
    transform[0, 3] = position_3d["x"]
    transform[1, 3] = position_3d["y"]
    transform[2, 3] = position_3d["z"]

    return torch.from_numpy(transform).float().to(device)


def parse_stations_json(json_data: dict[str, Any]) -> list[dict[str, Any]]:
    """Parse and validate the stations JSON structure.

    Args:
        json_data: Parsed JSON dict from the uploaded stations.json.

    Returns:
        List of station dicts, each containing at minimum:
        id, name, path_to_image, position_3d, orientation_3d.

    Raises:
        ValueError: If required fields are missing or malformed.
    """
    if "stations" not in json_data:
        raise ValueError("stations.json must contain a 'stations' array")

    stations = json_data["stations"]
    if not isinstance(stations, list) or len(stations) == 0:
        raise ValueError("'stations' must be a non-empty array")

    required_fields = ["id", "path_to_image", "position_3d", "orientation_3d"]
    for i, station in enumerate(stations):
        for field in required_fields:
            if field not in station:
                raise ValueError(
                    f"Station {i} ('{station.get('name', 'unknown')}') "
                    f"is missing required field '{field}'"
                )

        pos = station["position_3d"]
        if not all(k in pos for k in ("x", "y", "z")):
            raise ValueError(
                f"Station {i}: position_3d must have 'x', 'y', 'z' keys"
            )

        ori = station["orientation_3d"]
        if not all(k in ori for k in ("qx", "qy", "qz", "qw")):
            raise ValueError(
                f"Station {i}: orientation_3d must have 'qx', 'qy', 'qz', 'qw' keys"
            )

    return stations


def match_stations_to_files(
    stations: list[dict[str, Any]],
    uploaded_filenames: list[str],
) -> list[tuple[dict[str, Any], str]]:
    """Match uploaded image filenames to stations via basename of path_to_image.

    Args:
        stations: Parsed station dicts from parse_stations_json().
        uploaded_filenames: List of original filenames from the uploaded files.

    Returns:
        List of (station_dict, matched_uploaded_filename) tuples,
        ordered by the station's position in the JSON.

    Raises:
        ValueError: If any uploaded file cannot be matched to a station,
                    or if any station has no matching uploaded file.
    """
    # Build a lookup: basename -> uploaded filename
    filename_lookup: dict[str, str] = {}
    for fname in uploaded_filenames:
        basename = Path(fname).name
        filename_lookup[basename] = fname

    matched = []
    unmatched_stations = []

    for station in stations:
        # The path_to_image might be like "extracted_frames/frame_000046.179467.jpg"
        station_basename = Path(station["path_to_image"]).name

        if station_basename in filename_lookup:
            matched.append((station, filename_lookup[station_basename]))
        else:
            unmatched_stations.append(
                f"{station.get('name', station['id'])} "
                f"(expects '{station_basename}')"
            )

    if unmatched_stations:
        raise ValueError(
            f"Could not find uploaded images for {len(unmatched_stations)} station(s): "
            + ", ".join(unmatched_stations)
        )

    return matched
