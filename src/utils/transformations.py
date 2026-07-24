import numpy as np
import torch
import cv2

from src._datatypes import SliceTransducerGeometry
from src.trainings_params import Parameter_Demo2D_curvylinear


def standardize_volume(volume: np.ndarray, mean: float, mask: np.ndarray | None = None) -> np.ndarray:

    if mask is not None:
        volume_mask = mask
    else:
        volume_mask = volume > 0

    volume = volume - np.mean(volume[volume_mask])
    volume = volume / np.std(volume[volume_mask])
    volume = volume * (mean / 3)
    volume = volume + mean
    volume[volume < 0] = 0
    volume = volume.astype(np.float32)
    return volume


def rotmat_from_euler(thetas: torch.Tensor) -> torch.Tensor:
    """Gets 4X4 homogenious rotation matrices from euler angles.

    Args:
        thetas (torch.Tensor): Tensor containing n sets of eulerangels with rotations around x,y,z. Dimensions: (n,3)

    Returns:
        torch.tensor: retuns set of 4X4 rotations matrices. Dimensions: (n,3,3).
    """

    assert len(thetas.shape) == 2, "Most likly batch dimension missing."
    assert thetas.shape[1] == 3, "Can only handel 3D rotations but got {thetas.shape[1]}D."
    bs = thetas.shape[0]
    zero = torch.zeros((bs, 1), dtype=torch.float32, device=thetas.device)
    one = torch.ones((bs, 1), dtype=torch.float32, device=thetas.device)

    # rotations around x
    x = torch.cat(
        [
            one,
            zero,
            zero,
            zero,
            zero,
            torch.cos(thetas[:, 0:1]),
            -torch.sin(thetas[:, 0:1]),
            zero,
            zero,
            torch.sin(thetas[:, 0:1]),
            torch.cos(thetas[:, 0:1]),
            zero,
            zero,
            zero,
            zero,
            one,
        ],
        dim=1,
    ).reshape((bs, 4, 4))

    # rotations around y
    y = torch.cat(
        [
            torch.cos(thetas[:, 1:2]),
            zero,
            torch.sin(thetas[:, 1:2]),
            zero,
            zero,
            one,
            zero,
            zero,
            -torch.sin(thetas[:, 1:2]),
            zero,
            torch.cos(thetas[:, 1:2]),
            zero,
            zero,
            zero,
            zero,
            one,
        ],
        dim=1,
    ).reshape((bs, 4, 4))

    # rotations around z
    z = torch.cat(
        [
            torch.cos(thetas[:, 2:3]),
            -torch.sin(thetas[:, 2:3]),
            zero,
            zero,
            torch.sin(thetas[:, 2:3]),
            torch.cos(thetas[:, 2:3]),
            zero,
            zero,
            zero,
            zero,
            one,
            zero,
            zero,
            zero,
            zero,
            one,
        ],
        dim=1,
    ).reshape((bs, 4, 4))

    # combine
    Rs = torch.bmm(z, torch.bmm(y, x))

    return Rs


def transform_slice(img_slice: torch.Tensor, aff_trans_mats: torch.Tensor) -> torch.Tensor:
    """transforms coordinates of slice by applying affine transformation matrices

    Args:
        img_slice (torch.Tensor): _description_
        aff_trans_mats (torch.Tensor): _description_

    Returns:
        torch.Tensor: _description_
    """
    slice_shape = img_slice.shape

    assert img_slice.shape == torch.Size(
        (slice_shape[0], slice_shape[1], slice_shape[2], 3)
    ), "image slice needs to be of shape (batch size, fan radial range, fan angular range, 3)"
    assert aff_trans_mats.shape == torch.Size(
        (slice_shape[0], 4, 4)
    ), "The matrix arraz needs to be of shape (batch_size, 4, 4)."

    # get homogenious coordinates
    transformed = torch.ones((*slice_shape[:3], 4)).to(device=img_slice.device)
    transformed[:, :, :, :3].copy_(img_slice)

    # transform
    transformed = torch.einsum("bmnj, bij -> bmni", transformed, aff_trans_mats)

    return transformed[:, :, :, :3]


def resample_to_simulation_space(
    data: np.ndarray, geometries: list[SliceTransducerGeometry], params: Parameter_Demo2D_curvylinear
) -> np.ndarray:
    polar_shape = params.image_size_polar
    resampled = []
    for i, geometry in enumerate(geometries):
        slice_cart = np.asarray(data[..., i], dtype=np.float32)
        map_x, map_y = _build_simulation_to_image_maps(geometry, polar_shape)
        resampled_slice = cv2.remap(
            slice_cart,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        resampled.append(resampled_slice)
    return np.transpose(np.asarray(resampled), (1, 2, 0))  # reshape from (n, h, w) to (h, w, n)


def resample_to_image_space(
    data: np.ndarray, geometries: list[SliceTransducerGeometry], params: Parameter_Demo2D_curvylinear
) -> np.ndarray:
    cart_shape = tuple(params.get("image_size_cart", _infer_cart_shape_from_geometries(geometries)))
    resampled = []
    for i, geometry in enumerate(geometries):
        slice_sim = np.asarray(data[..., i], dtype=np.float32)
        map_x, map_y = _build_image_to_simulation_maps(geometry, cart_shape, slice_sim.shape)
        resampled_slice = cv2.remap(
            slice_sim,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        resampled.append(resampled_slice)
    return np.transpose(np.asarray(resampled), (1, 2, 0))


def _build_simulation_to_image_maps(geometry: SliceTransducerGeometry, polar_shape):
    num_samples_per_scanline, num_scan_lines = polar_shape
    source_pix, r_min, r_max, phi_start, phi_delta = _compute_geometry_parameters(geometry)

    radii = np.linspace(r_min, r_max, num_samples_per_scanline, dtype=np.float32)
    angles = np.linspace(phi_start, phi_start + phi_delta, num_scan_lines, dtype=np.float32)
    radii_grid, angles_grid = np.meshgrid(radii, angles, indexing="ij")

    map_y = source_pix[0] + radii_grid * np.sin(angles_grid)
    map_x = source_pix[1] + radii_grid * np.cos(angles_grid)

    return map_x.astype(np.float32, copy=False), map_y.astype(np.float32, copy=False)


def _build_image_to_simulation_maps(
    geometry: SliceTransducerGeometry, cart_shape: tuple[int, int], polar_shape: tuple[int, int]
):
    height, width = cart_shape
    num_samples_per_scanline, num_scan_lines = polar_shape
    source_pix, r_min, r_max, phi_start, phi_delta = _compute_geometry_parameters(geometry)

    grid_y, grid_x = np.meshgrid(
        np.arange(height, dtype=np.float32),
        np.arange(width, dtype=np.float32),
        indexing="ij",
    )
    rel_y = grid_y - source_pix[0]
    rel_x = grid_x - source_pix[1]
    radii = np.sqrt(rel_y * rel_y + rel_x * rel_x, dtype=np.float32)
    angles = np.arctan2(rel_y, rel_x)
    angle_offset = ((angles - phi_start + np.pi) % (2 * np.pi)) - np.pi

    if np.isclose(r_max, r_min):
        map_y = np.zeros_like(radii, dtype=np.float32)
    else:
        map_y = (radii - r_min) * ((num_samples_per_scanline - 1) / (r_max - r_min))

    if np.isclose(phi_delta, 0.0):
        map_x = np.zeros_like(angles, dtype=np.float32)
    else:
        map_x = angle_offset * ((num_scan_lines - 1) / phi_delta)

    outside_mask = (radii < r_min) | (radii > r_max) | (map_x < 0) | (map_x > (num_scan_lines - 1))
    map_x = map_x.astype(np.float32, copy=False)
    map_y = map_y.astype(np.float32, copy=False)
    map_x[outside_mask] = -1.0
    map_y[outside_mask] = -1.0
    return map_x, map_y


@staticmethod
def _compute_geometry_parameters(geometry: SliceTransducerGeometry):
    source_pix = np.asarray(geometry.source_pix, dtype=np.float32)
    ang_limits = np.asarray(geometry.angle_limit_points_pix, dtype=np.float32)
    r_range = np.asarray(geometry.radius_range_mm, dtype=np.float32)

    phi_start = float(np.arctan2(ang_limits[0, 0] - source_pix[0], ang_limits[0, 1] - source_pix[1]))
    phi_end = float(np.arctan2(ang_limits[1, 0] - source_pix[0], ang_limits[1, 1] - source_pix[1]))
    phi_delta = ((phi_end - phi_start + np.pi) % (2 * np.pi)) - np.pi
    if np.isclose(phi_delta, 0.0) and not np.isclose(phi_start, phi_end):
        phi_delta = 2 * np.pi

    return source_pix, float(r_range[0]), float(r_range[1]), phi_start, float(phi_delta)


@staticmethod
def _infer_cart_shape_from_geometries(geometries: list[SliceTransducerGeometry]) -> tuple[int, int]:
    max_y = 0.0
    max_x = 0.0
    for geometry in geometries:
        source_pix = np.asarray(geometry.source_pix, dtype=np.float32)
        ang_limits = np.asarray(geometry.angle_limit_points_pix, dtype=np.float32)
        r_max = float(np.max(np.asarray(geometry.radius_range_mm, dtype=np.float32)))
        phi_start = float(np.arctan2(ang_limits[0, 0] - source_pix[0], ang_limits[0, 1] - source_pix[1]))
        phi_end = float(np.arctan2(ang_limits[1, 0] - source_pix[0], ang_limits[1, 1] - source_pix[1]))
        phi_delta = ((phi_end - phi_start + np.pi) % (2 * np.pi)) - np.pi
        if np.isclose(phi_delta, 0.0) and not np.isclose(phi_start, phi_end):
            phi_delta = 2 * np.pi
        angles = np.linspace(phi_start, phi_start + phi_delta, 128, dtype=np.float32)
        arc_y = source_pix[0] + r_max * np.sin(angles)
        arc_x = source_pix[1] + r_max * np.cos(angles)
        max_y = max(max_y, float(source_pix[0]), float(np.max(arc_y)), float(np.max(ang_limits[:, 0])))
        max_x = max(max_x, float(source_pix[1]), float(np.max(arc_x)), float(np.max(ang_limits[:, 1])))
    return int(np.ceil(max_y)) + 1, int(np.ceil(max_x)) + 1
