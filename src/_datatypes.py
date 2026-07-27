"""------------------------------------------------------------------------------
RFlash - Official implementation of the RFlash framework
Author:
    Valentin Bacher
    valentin.bacher@cs.ox.ac.uk
Affiliation:
    OMNI Lab
    Department of Computer Science
    University of Oxford
    https://omni.cs.ox.ac.uk/
Purpose:
    Immutable data containers for estimated ultrasound transducer geometry.
License:
    This file is part of the RFlash project and is distributed under the
    repository's LICENSE. See the LICENSE file in the repository root for
    licensing information.
------------------------------------------------------------------------------"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SliceTransducerGeometry:
    """Geometry estimate for one 2D curvilinear ultrasound slice.

    Attributes:
        source_pix: Virtual acoustic source position in image pixels as
            ``(row, column)``.
        source_mm: Virtual acoustic source position in millimetres.
        radius_range_mm: Inner and outer fan radii in millimetres.
        angular_range_rad: Fan opening angle in radians.
        angular_range_deg: Fan opening angle in degrees.
        angle_limit_points_pix: Two lower fan-boundary points in pixels.
    """

    source_pix: np.ndarray
    source_mm: np.ndarray
    radius_range_mm: np.ndarray
    angular_range_rad: float
    angular_range_deg: float
    angle_limit_points_pix: list[np.ndarray]


@dataclass(frozen=True)
class VolumeTransducerGeometry:
    """Geometry estimate for a 3D ultrasound volume.

    The fetal-brain demo models the probe geometry with a fan direction and an
    orthogonal tilt direction. Each direction stores the estimated virtual
    source, radial support, and angular support.

    Attributes:
        fan_source_pix: Fan source position in volume pixels.
        fan_source_mm: Fan source position in millimetres.
        fan_r_range_mm: Fan radial range in millimetres.
        fan_ang_range_rad: Fan angular range in radians.
        fan_ang_range_deg: Fan angular range in degrees.
        tilt_source_pix: Tilt source position in volume pixels.
        tilt_source_mm: Tilt source position in millimetres.
        tilt_r_range_mm: Tilt radial range in millimetres.
        tilt_ang_range_rad: Tilt angular range in radians.
        tilt_ang_range_deg: Tilt angular range in degrees.
    """

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
