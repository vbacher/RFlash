"""Estimate the fan geometry of curvilinear ultrasound images.

The public demo only needs the scanner geometry implied by a normal 2D fan
shape: two approximately straight lateral fan boundaries that meet at the
virtual acoustic point source.  This module therefore avoids OpenCV/Hough
dependencies and fits those two boundaries directly from the non-empty image
support.
"""

from __future__ import annotations

from dataclasses import dataclass
import sys
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from sys import stderr
from cv2 import Canny, HoughLinesP

# project imports
from src._datatypes import SliceTransducerGeometry, VolumeTransducerGeometry

# FIXME: remove this import
from src.utils import visualize_2d_image


def get_intersect(points: np.ndarray) -> tuple[float, float]:
    """calculates the intersection of two lines parametrized by points.

    Args:
        points (np.ndarray): Array of shape (4,2) containing both endpoints (x_1,x_2) of both lines (l1, l2), i.e.
        [x_1_l1, x_2_l1, x_1_l2, x_2_l2]

    Returns:
        tuple[float,float]: Point of intersection (x_c, y_c)

    Example:
        >>> points = np.asarray([[0,1],[0,2],[1,0],[2,0]])
        >>> x_c, y_c = __get_intersect(points)
        >>> assert x_c == 0 and y_c == 0
    """
    s = np.vstack([points[0], points[1], points[2], points[3]])  # s for stacked
    h = np.hstack((s, np.ones((4, 1))))  # h for homogeneous
    l1 = np.cross(h[0], h[1])  # get first line
    l2 = np.cross(h[2], h[3])  # get second line
    x, y, z = np.cross(l1, l2)  # point of intersection
    if z == 0:  # lines are parallel
        return (float("inf"), float("inf"))
    return (x / z, y / z)


def __angle_between_points(center_point: np.ndarray, leg_point_a: np.ndarray, leg_point_b: np.ndarray) -> float:
    """Helper function. Returns angle between three points based at center

    Args:
        center_point (np.ndarray): Coordinates of centering point
        leg_point_a (np.ndarray): Coordinates of first leg
        leg_point_b (np.ndarray): Coordinates of second leg

    Returns:
        float: resulting angle in radiants

    Example:
        >>> a = [0,1]
        >>> b = [1,0]
        >>> c = [0,0]
        >>> ang = __angle_between_points(c,a,b)
        >>> assert np.pi/2 -0.001 <= ang <= np.pi/2 + 0.001
    """
    ca = leg_point_a - center_point
    cb = leg_point_b - center_point

    divisor = np.linalg.norm(ca) * np.linalg.norm(cb)
    if divisor == 0:
        print("two points are identical", file=stderr)
        return None
    elif divisor == float("inf"):
        return 0.0
    cosine_angle = np.dot(ca, cb) / divisor
    return np.arccos(cosine_angle)


def _as_uint8_image(image: np.ndarray) -> np.ndarray:
    """Normalize a 2D image to uint8 for display."""

    image_array = np.asarray(image, dtype=float)
    finite_mask = np.isfinite(image_array)
    if not finite_mask.any():
        return np.zeros(image_array.shape, dtype=np.uint8)

    finite_values = image_array[finite_mask]
    image_min = float(finite_values.min())
    image_max = float(finite_values.max())
    if np.isclose(image_min, image_max):
        return np.zeros(image_array.shape, dtype=np.uint8)

    clipped = np.nan_to_num(image_array, nan=image_min, posinf=image_max, neginf=image_min)
    return np.clip((clipped - image_min) / (image_max - image_min) * 255.0, 0, 255).astype(np.uint8)


def visualize_geometry_overlay(
    image: np.ndarray,
    geometry: SliceTransducerGeometry,
    output_path: str | Path | None = None,
    show: bool = True,
) -> Path | None:
    """Draw the estimated source, boundary lines, and fan sector over a slice.

    Args:
        image: 2D ultrasound slice.
        spacing_plane_mm: Pixel spacing as ``(row_spacing, column_spacing)``.
        geometry: Geometry estimate returned by :func:`estimate_geometry_slice`.
        output_path: Optional path for a saved PNG overlay.
        show: Whether to open a matplotlib window.

    Returns:
        The saved output path when ``output_path`` is provided.
    """
    height, width = image.shape
    image_u8 = _as_uint8_image(image)
    spacing_mm = geometry.source_mm / geometry.source_pix
    source_mm = np.asarray(geometry.source_mm, dtype=float)
    source_row, source_col = source_mm
    left_point, right_point = (
        np.asarray(point, dtype=float) * spacing_mm for point in geometry.angle_limit_points_pix
    )
    r_inner_mm, r_outer_mm = geometry.radius_range_mm

    fig, ax = plt.subplots(figsize=(9, 7))
    extent = [0, width * spacing_mm[1], height * spacing_mm[0], 0]
    ax.imshow(image_u8, cmap="gray", origin="upper", extent=extent)

    ax.scatter([source_col], [source_row], marker="+", s=140, c="red", linewidths=2.0, label="source")
    ax.plot(
        [source_col, left_point[1], np.nan, source_col, right_point[1]],
        [source_row, left_point[0], np.nan, source_row, right_point[0]],
        color="tab:cyan",
        linewidth=1.8,
    )

    num_samples = 100
    vec = (left_point - source_mm) / np.linalg.norm(left_point - source_mm)
    start_point_1 = source_mm + r_inner_mm * vec
    start_point_2 = source_mm + r_outer_mm * vec
    start_point_1_delta = start_point_1 - source_mm
    start_point_1_polar = (
        float(np.linalg.norm(start_point_1_delta)),
        float(np.arctan2(start_point_1_delta[0], start_point_1_delta[1])),
    )
    start_point_2_delta = start_point_2 - source_mm
    start_point_2_polar = (
        float(np.linalg.norm(start_point_2_delta)),
        float(np.arctan2(start_point_2_delta[0], start_point_2_delta[1])),
    )
    right_point_delta = right_point - source_mm
    theta_right = float(np.arctan2(right_point_delta[0], right_point_delta[1]))
    theta_delta = ((theta_right - start_point_1_polar[1] + np.pi) % (2 * np.pi)) - np.pi
    line_1_polar = np.linspace(start_point_1_polar[1], start_point_1_polar[1] + theta_delta, num_samples)
    line_2_polar = np.linspace(start_point_2_polar[1], start_point_2_polar[1] + theta_delta, num_samples)
    line_1_polar = np.vstack([np.full_like(line_1_polar, r_inner_mm), line_1_polar])
    line_2_polar = np.vstack([np.full_like(line_2_polar, r_outer_mm), line_2_polar])

    line_1_cartesian = np.vstack(
        [
            source_mm[0] + line_1_polar[0] * np.sin(line_1_polar[1]),
            source_mm[1] + line_1_polar[0] * np.cos(line_1_polar[1]),
        ]
    )
    line_2_cartesian = np.vstack(
        [
            source_mm[0] + line_2_polar[0] * np.sin(line_2_polar[1]),
            source_mm[1] + line_2_polar[0] * np.cos(line_2_polar[1]),
        ]
    )
    ax.plot(line_1_cartesian[1], line_1_cartesian[0], color="yellow", linewidth=1.0)
    ax.plot(line_2_cartesian[1], line_2_cartesian[0], color="yellow", linewidth=1.0)

    ax.set_title(
        "Estimated scanner geometry: "
        f"source=({source_row:.1f}, {source_col:.1f}) mm, "
        f"angle={geometry.angular_range_deg:.1f} deg"
    )
    ax.set_xlabel("column [mm]")
    ax.set_ylabel("row [mm]")
    ax.set_aspect("equal")

    saved_path = None
    if output_path is not None:
        saved_path = Path(output_path)
        saved_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(saved_path, dpi=160, bbox_inches="tight")

    if show:
        plt.show()
    plt.close(fig)
    return saved_path


def _confirm_estimate(prompt: str) -> bool:
    """Ask the user whether a displayed geometry estimate is acceptable."""

    if not sys.stdin.isatty():
        return True

    while True:
        response = input(f"{prompt} [Y/n]: ").strip().lower()
        if response in ("", "y", "yes"):
            return True
        if response in ("n", "no"):
            return False
        print("Please answer 'y' or 'n'.")


def estimate_geometry_slice(
    plane: np.ndarray,
    spacing_plane_mm: Iterable[float] = (1.0, 1.0),
    overlay_path: str | Path | None = None,
    verbose: bool = False,
) -> tuple[SliceTransducerGeometry | None, bool]:
    """Estimate the acoustic source and fan limits for one 2D fan slice.

    Args:
        plane: 2D ultrasound image with a normal curvilinear fan shape.
        spacing_plane_mm: Pixel spacing as ``(row_spacing, column_spacing)``.
        threshold: Optional intensity threshold for separating image support
            from background. When omitted, Otsu's threshold is computed.
        confirm: If true and stdin is interactive, show the overlay and ask the
            user whether the estimate is acceptable.
        overlay_path: Optional path where the overlay should be saved.
        show_overlay: Whether to display the overlay with matplotlib.

    Returns:
        A :class:`SliceGeometry` estimate.

    Raises:
        ValueError: If the slice is not 2D or the fan boundaries cannot be fit.
    """

    plane_array = np.asarray(plane)
    if plane_array.ndim != 2:
        raise ValueError(f"Expected a 2D slice, got shape {plane_array.shape}.")

    spacing = np.asarray(tuple(spacing_plane_mm), dtype=float)
    if spacing.shape != (2,):
        raise ValueError("spacing_plane_mm must contain row and column spacing.")

    support = (plane_array > 0).astype(np.uint8) * 255  # binary mask of non-zero pixels

    # Canny edge detection
    plane_edges = Canny(support, 0, 3, None, 3)

    lines_plane = None
    threshold = int(min(plane.shape) * 0.8)

    while lines_plane is None and threshold > 0:
        # extract lines with Hughes transform
        lines_plane = HoughLinesP(
            plane_edges,
            1,
            np.pi / 180,
            threshold,
            None,
            int(min(plane.shape) * 0.008),
            int(min(plane.shape) * 0.08),
        )
        threshold -= 5

        # chack if more than 2 lines are detected.
        if lines_plane is not None:
            if lines_plane.shape[0] < 2:
                lines_plane = None
                continue
        else:
            continue

        # get longest two lines
        lines_vec = np.concatenate(
            [
                [lines_plane[..., 0] - lines_plane[..., 2]],
                [lines_plane[..., 1] - lines_plane[..., 3]],
            ]
        ).T
        lines_length = np.sqrt(lines_vec[..., 0] ** 2 + lines_vec[..., 1] ** 2)

        # ignore (almost) vertical and horizontal lines
        mask = np.logical_and(
            np.abs(lines_vec[..., 0]) > int(plane.shape[0] * 0.005),
            np.abs(lines_vec[..., 1]) > int(plane.shape[1] * 0.005),
        )
        try:
            mask_vertical = np.abs(lines_vec[:, 0]) < int(plane.shape[0] * 0.02)
            center_line = plane.shape[1] // 2
        except:
            pass

        # get longest ascending and descending lines
        mask_ascending = np.logical_and(mask, lines_vec[..., 1] > 0)
        mask_descending = np.logical_and(mask, lines_vec[..., 1] < 0)
        if sum(mask_ascending) < 1 or sum(mask_descending) < 1:
            lines_plane = None
            continue
        try:
            lines_plane = np.concatenate(
                [
                    lines_plane[mask_descending][np.argmax(lines_length[mask_descending])][None, ...],
                    lines_plane[mask_ascending][np.argmax(lines_length[mask_ascending])][None, ...],
                ]
            )
        except:
            print("None")

    assert lines_plane.__len__() == 2, f"edge finding failed. Instead of 2, {lines_plane.__len__()} were found."
    # find source by intersecting lines
    x_2_pix, x_1_pix = get_intersect(lines_plane.reshape(4, 2))

    # get coords of point source
    source_pix = np.asarray([x_1_pix, x_2_pix])
    source_mm = source_pix * spacing_plane_mm

    # get_r_min
    highest_point = np.min(np.arange(support.shape[0])[np.max(support, axis=1) > 0])
    p_1_mm = (
        np.asarray(
            get_intersect(
                np.concatenate(
                    [
                        np.asarray([lines_plane[0], [0, highest_point, support.shape[1], highest_point]]).reshape(
                            4, 2
                        ),
                        [[highest_point, 0], [highest_point, support.shape[1]]],
                    ]
                )
            )
        )[::-1]
        * spacing_plane_mm
    )
    p_2_mm = (
        np.asarray(
            get_intersect(
                np.concatenate(
                    [
                        np.asarray([lines_plane[1], [0, highest_point, support.shape[1], highest_point]]).reshape(
                            4, 2
                        ),
                        [[highest_point, 0], [highest_point, support.shape[1]]],
                    ]
                )
            )
        )[::-1]
        * spacing_plane_mm
    )
    r_min_mm = max(np.linalg.norm(p_1_mm - source_mm), np.linalg.norm(p_2_mm - source_mm))

    # get_r_max
    lowest_point = np.max(np.arange(support.shape[0])[np.max(support, axis=1) > 0])
    max_extend_of_loest_points = np.where(support[lowest_point] > 0)[0]
    p_1_mm = np.asarray([lowest_point, max_extend_of_loest_points[0]]) * spacing_plane_mm
    p_2_mm = np.asarray([lowest_point, max_extend_of_loest_points[-1]]) * spacing_plane_mm
    r_max_mm = max(np.linalg.norm(p_1_mm - source_mm), np.linalg.norm(p_2_mm - source_mm))

    # get angular range
    angular_range_rad = __angle_between_points(
        source_mm,
        lines_plane[0, 4:1:-1] * spacing_plane_mm,
        lines_plane[1, 4:1:-1] * spacing_plane_mm,
    )
    if angular_range_rad is None:
        return None
    angular_range_deg = np.degrees(angular_range_rad)

    # get angular limits
    ## get four endpoints
    points = np.concatenate(lines_plane.reshape(2, 2, 2), axis=0)[:, ::-1]
    # get two points with largest y value (lower two points)
    ang_limits_pix = points[np.lexsort((points[:, 1], points[:, 0]))][2:]
    ang_limits_pix = ang_limits_pix[np.argsort(ang_limits_pix[:, 0])][::-1]

    geometry = SliceTransducerGeometry(
        source_pix=source_pix,
        source_mm=source_mm,
        radius_range_mm=np.asarray([r_min_mm, r_max_mm]),
        angular_range_rad=angular_range_rad,
        angular_range_deg=angular_range_deg,
        angle_limit_points_pix=ang_limits_pix,
    )

    if verbose or overlay_path is not None:
        visualize_geometry_overlay(plane, geometry, output_path=overlay_path, show=verbose)
    if verbose:
        ok = _confirm_estimate(f"\nPlease check {overlay_path}.\nWas the scanner source found correctly?")
        if not ok:
            return None, False
        # if user confirmed and an overlay image was written, remove it
        if overlay_path is not None:
            try:
                p = Path(overlay_path)
                if p.exists():
                    p.unlink()
            except Exception:
                pass

    return geometry, True


def estimate_scanner_geometry_volume(
    volume: np.ndarray,
    spacing_mm: Iterable[float] = (1.0, 1.0, 1.0),
    overlay_dir: str | Path | None = None,
    verbose: bool = False,
) -> VolumeTransducerGeometry:
    """Estimate scanner geometry from the two middle planes of a 3D volume.

    The fetal demo volume is treated as a stack of normal 2D fan slices.  The
    source is estimated once in the coronal middle plane and once in the axial
    middle plane, matching the original research code's ``fan`` and ``tilt``
    convention.
    """

    volume_array = np.asarray(volume)
    if volume_array.ndim != 3:
        raise ValueError(f"Expected a 3D volume, got shape {volume_array.shape}.")

    spacing = np.asarray(tuple(spacing_mm), dtype=float)
    if spacing.shape != (3,):
        raise ValueError("spacing_mm must contain three values.")

    # Find middle planes
    middle = [axis_size // 2 for axis_size in volume_array.shape]

    # get overlay paths
    overlay_root = Path(overlay_dir) if overlay_dir is not None else None
    coronal_overlay = overlay_root / "geometry_coronal_middle.png" if overlay_root else None
    axial_overlay = overlay_root / "geometry_axial_middle.png" if overlay_root else None

    # Estimate geometry in the coronal middle planes.
    ## transpose to get transducer on top
    coronal_midplane = np.transpose(volume[middle[0]])  # y-z
    coronal_spacing_mm = spacing_mm[1:][::-1]

    coronal_geometry, coronal_success = estimate_geometry_slice(
        coronal_midplane,
        spacing_plane_mm=coronal_spacing_mm,
        overlay_path=coronal_overlay,
        verbose=verbose,
    )

    # Estimate geometry in the axial middle planes.
    ## transpose to get transducer on top
    axial_midplane = np.transpose(volume[:, middle[1]])  # x-z
    axial_spacing_mm = spacing[::-2]
    axial_geometry, axial_success = estimate_geometry_slice(
        axial_midplane, spacing_plane_mm=axial_spacing_mm, overlay_path=axial_overlay, verbose=verbose
    )
    if not coronal_success or not axial_success:
        raise ValueError("Failed to estimate scanner geometry in one or both middle planes.")

    vol_trans_geometry = VolumeTransducerGeometry(
        fan_source_pix=np.asarray([axial_geometry.source_pix[1], middle[1], axial_geometry.source_pix[0]]),
        fan_source_mm=np.asarray([axial_geometry.source_mm[1], middle[1] * spacing[1], axial_geometry.source_mm[0]]),
        fan_r_range_mm=axial_geometry.radius_range_mm,
        fan_ang_range_rad=axial_geometry.angular_range_rad,
        fan_ang_range_deg=axial_geometry.angular_range_deg,
        tilt_source_pix=np.asarray([middle[0], coronal_geometry.source_pix[1], coronal_geometry.source_pix[0]]),
        tilt_source_mm=np.asarray(
            [middle[0] * spacing[0], coronal_geometry.source_mm[1], coronal_geometry.source_mm[0]]
        ),
        tilt_r_range_mm=coronal_geometry.radius_range_mm,
        tilt_ang_range_rad=coronal_geometry.angular_range_rad,
        tilt_ang_range_deg=coronal_geometry.angular_range_deg,
    )

    return vol_trans_geometry


def _get_number_of_slices_to_process(num_files: int) -> int:
    print(
        "\nThe estimation of the scanner geometry is not alywas stable, especially if the edges of the fan are not clearly visible.\n"
        f"The current selection contains {num_files} slices. You have to manually check the found geometry for every slice.\n"
    )
    num_slices = -1
    while num_slices < 1:
        i_val = input("Please select how many slices you want to process: (Press ENTER to process all slices)")
        if i_val == "":
            num_slices = num_files
        else:
            try:
                num_slices = int(i_val)
                if num_slices < 1 or num_slices > num_files:
                    print(f"\nPlease enter a number between 1 and {num_files}.\n")
                    num_slices = -1
            except ValueError:
                print("\nPlease enter a valid integer.\n")
                num_slices = -1
    return num_slices


def estimate_scanner_geometries_stack(
    stack: np.ndarray,
    spacing_mm: Iterable[float] = (1.0, 1.0, 1.0),
    overlay_dir: str | Path | None = None,
    verbose: bool = False,
) -> StackTransducerGeometry:
    """Estimate scanner geometry for each slice in a stack of 2D images.

    Args:
        stack: 3D array of shape (num_slices, height, width).
        spacing_mm: Pixel spacing as ``(row_spacing, column_spacing)``.
        overlay_dir: Optional directory where overlays should be saved.
        verbose: Whether to display overlays with matplotlib.

    Returns:
        A list of :class:`SliceTransducerGeometry` estimates for each slice.
    """

    stack_array = np.asarray(stack)
    if stack_array.ndim != 3:
        raise ValueError(f"Expected a 3D stack, got shape {stack_array.shape}.")

    spacing = np.asarray(tuple(spacing_mm), dtype=float)
    if spacing.shape != (2,):
        raise ValueError("spacing_mm must contain two values.")

    num_slices = _get_number_of_slices_to_process(stack_array.shape[0])

    geometries = []
    random_selection = np.random.choice(stack_array.shape[0], size=stack_array.shape[0], replace=False)
    successful_indices = []
    for i in random_selection:
        slice_image = stack_array[i]
        overlay_path = Path(overlay_dir) / f"geometry_slice.png" if overlay_dir else None
        geometry, success = estimate_geometry_slice(
            slice_image,
            spacing_plane_mm=spacing,
            overlay_path=overlay_path,
            verbose=verbose,
        )
        if success:
            geometries.append(geometry)
            successful_indices.append(i)

        if len(successful_indices) >= num_slices:
            break

    return geometries, successful_indices
