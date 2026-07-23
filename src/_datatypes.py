import numpy as np
from dataclasses import dataclass


@dataclass(frozen=True)
class SliceTransducerGeometry:
    source_pix: np.ndarray
    source_mm: np.ndarray
    radius_range_mm: np.ndarray
    angular_range_rad: float
    angular_range_deg: float
    angle_limit_points_pix: list[np.ndarray]


@dataclass(frozen=True)
class VolumeTransducerGeometry:
    fan_source_pix: np.ndarray
    fan_source_mm: np.ndarray
    fan_r_range_mm: np.ndarray
    fan_ang_range_rad: float
    fan_ang_range_deg: float
    tilt_source_pix: np.ndarray
    tilt_source_mm: np.ndarray
    tilt_r_range_mm: np.ndarray
    tilt_ang_range_rad: float
    tilt_ang_range_deg: float
