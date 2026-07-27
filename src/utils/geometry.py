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
    Geometry handlers and interpolation kernels for sampling ultrasound slices
    from dense volumes or aligned image stacks.
License:
    This file is part of the RFlash project and is distributed under the
    repository's LICENSE. See the LICENSE file in the repository root for
    licensing information.
------------------------------------------------------------------------------"""

import sys

import numpy as np
import torch

from src.utils.transformations import transform_slice
from src.utils.visualization import visualize_vol


class VolumeHandler:
    """Hold a 3D volume, its mask, and interpolation methods."""

    def __init__(
        self,
        volume: torch.Tensor,
        pixel_spacing: list[float] | np.ndarray = [1, 1, 1],
        volume_mask: torch.Tensor | None = None,
    ):
        """Construct a volume handler.

        Args:
            volume: 3D tensor with shape ``(height, width, depth)``.
            pixel_spacing: Voxel spacing in millimetres.
            volume_mask: Optional foreground mask with the same shape as
                ``volume``. If omitted, positive voxels are foreground.

        Raises:
            AssertionError: If the volume is not 3D or spacing does not have
                three entries.
        """

        assert volume.ndim == 3, "Volume tensor must be 3D"
        assert volume.ndim == pixel_spacing.__len__(), "Pixel spacing must have size 3."

        # The public demo uses zero-valued background outside the ultrasound
        # fan, so positive values define the default support mask.
        if volume_mask is None:
            volume_mask = volume.clone()
            volume_mask[volume_mask > 0] = 1
            volume_mask[volume_mask < 0] = 0
        self.vol_mask = volume_mask

        self.shape = volume.shape

        self.volume = volume
        self.pixel_spacing = pixel_spacing

    def get_pixel_value_cart(self, coord_array: torch.Tensor) -> torch.Tensor:
        """Return volume intensities at Cartesian slice coordinates.

        Args:
            coord_array: Coordinates with shape ``(batch, height, width, 3)``.

        Returns:
            Interpolated intensities with shape ``(batch, height, width)``.
        """
        return trilinear_interpolation(coord_array, self.volume[..., None])[..., -1]

    def get_mask_value_cart(self, coord_array: torch.Tensor) -> torch.Tensor:
        """Return mask values at Cartesian slice coordinates.

        Args:
            coord_array: Coordinates with shape ``(batch, height, width, 3)``.

        Returns:
            Boolean mask with shape ``(batch, height, width)``.
        """
        interpolation = trilinear_interpolation(coord_array, self.vol_mask[..., None])[..., -1]
        return interpolation > 0

    def visualize_volume(self) -> None:
        """Visualize the three central planes of the stored volume."""
        visualize_vol(self.volume, self.pixel_spacing)

    def visualize_mask(self) -> None:
        """Visualize the three central planes of the stored volume mask."""
        visualize_vol(self.vol_mask, self.pixel_spacing)


class StackHandler:
    """Hold an aligned 2D image stack and bilinear interpolation methods."""

    def __init__(
        self,
        stack: torch.Tensor,
        pixel_spacing: list[float] = [1, 1, 1],
        volume_mask: torch.Tensor | None = None,
    ):
        """Construct a stack handler.

        Args:
            stack: 3D tensor with shape ``(height, width, num_slices)``.
            pixel_spacing: Spacing in millimetres for row, column, and slice axes.
            volume_mask: Optional foreground mask with the same shape as
                ``stack``. If omitted, positive pixels are foreground.

        Raises:
            AssertionError: If the stack is not 3D or spacing does not have
                three entries.
        """

        assert stack.ndim == 3, "Volume tensor must be 3D"
        assert stack.ndim == pixel_spacing.__len__(), "Pixel spacing must have size 3."

        # The stack demos use full-image masks by default, while this fallback
        # keeps the handler usable for zero-background stacks.
        if volume_mask is None:
            volume_mask = stack.clone()
            volume_mask[volume_mask > 0] = 1
            volume_mask[volume_mask < 0] = 0
        self.vol_mask = volume_mask

        self.shape = stack.shape

        self.volume = stack
        self.pixel_spacing = pixel_spacing

    def get_pixel_value_cart(self, coord_array: torch.Tensor) -> torch.Tensor:
        """Return stack intensities at Cartesian slice coordinates.

        Args:
            coord_array: Coordinates with shape ``(batch, height, width, 3)``.

        Returns:
            Interpolated intensities with shape ``(batch, height, width)``.
        """
        return bilinear_interpolation_stack(coord_array, self.volume[..., None])[..., -1]

    def get_mask_value_cart(self, coord_array: torch.Tensor) -> torch.Tensor:
        """Return mask values at Cartesian slice coordinates.

        Args:
            coord_array: Coordinates with shape ``(batch, height, width, 3)``.

        Returns:
            Boolean mask with shape ``(batch, height, width)``.
        """
        interpolation = bilinear_interpolation_stack(coord_array, self.vol_mask[..., None])[..., -1]
        return interpolation > 0

    def visualize_stack(self) -> None:
        """Visualize the three central planes of the stored stack."""
        visualize_vol(self.volume, self.pixel_spacing)

    def visualize_mask(self) -> None:
        """Visualize the three central planes of the stored stack mask."""
        visualize_vol(self.vol_mask, self.pixel_spacing)


class SliceHandler:
    """Generate curvilinear fan-slice coordinate grids."""

    def __init__(
        self,
        frame_size_cart_pix: np.ndarray,  # x,y in pix
        frame_size_polar: np.ndarray | None,  # r, phi
        source_pos_slice_pix: np.ndarray,  # x,y
        source_pos_tilt_pix: np.ndarray | None = None,  # x,y
        pixel_spacing_cart: np.ndarray | list[float] = [1.0, 1.0],  # x,y
        radius_range_mm: np.ndarray | None = None,  # min, max
        angular_range: float | None = None,  # fan_angle
        angular_limits: list | None = None,  # limit points
        axis_added_dimension: int = 1,
    ) -> None:
        """Construct a curvilinear fan-slice handler.

        Args:
            frame_size_cart_pix: Cartesian frame size in pixels as
                ``(height, width)``.
            frame_size_polar: Fan discretization as
                ``(radial_samples, angular_samples)``.
            source_pos_slice_pix: Virtual slice source in pixels.
            source_pos_tilt_pix: Pivot source for the volume tilt dimension.
                If omitted, the slice source is used for backwards compatibility.
            pixel_spacing_cart: Pixel spacing in millimetres.
            radius_range_mm: Optional inner/outer fan radii in millimetres.
            angular_range: Optional fan opening angle in radians.
            angular_limits: Optional two fan boundary points in pixels.
            axis_added_dimension: Axis where the slice plane is embedded into
                3D coordinates.

        Raises:
            AssertionError: If ``frame_size_polar`` is missing.
        """

        assert frame_size_polar is not None, "frame_size_polar must be specified."

        # The cone is usually spanned by a curved 1D piezo crystal array, which
        # is tilted inside the 3D scanner head. ``source_pos_slice_pix`` denotes
        # the fan point source, while ``source_pos_tilt_pix`` denotes the pivot
        # around which the scanner is turned.
        if source_pos_tilt_pix is None:
            self.source_pos_tilt_pix = np.asarray(source_pos_slice_pix)
            print(
                "not specifying source_pos_pivot is depricated. please specity explicitly",
                file=sys.stderr,
            )
        else:
            self.source_pos_tilt_pix = np.asarray(source_pos_tilt_pix)
        self.source_pos_tilt_mm = self.source_pos_tilt_pix * pixel_spacing_cart

        self.frame_size_cart_pix = frame_size_cart_pix  # (H,W)
        self.frame_size_cart_mm = np.asarray(self.frame_size_cart_pix) * pixel_spacing_cart
        self.frame_size_pol = frame_size_polar
        self.source_pos_slice_pix = np.asarray(source_pos_slice_pix)  # relative to frame 0,0
        self.source_pos_slice_mm = self.source_pos_slice_pix * pixel_spacing_cart
        self.pixel_spacing = pixel_spacing_cart

        # If no explicit angular support is supplied, infer a broad default fan
        # from the image extent. Otherwise use the estimated fan limits.
        if angular_range is None and angular_limits is None:
            _, phi_1 = self.get_pol_from_cart(
                torch.tensor(0) - 0.3 * self.frame_size_cart_mm[0],
                torch.tensor(self.frame_size_cart_mm[1]),
            )
            _, phi_2 = self.get_pol_from_cart(
                torch.tensor(self.frame_size_cart_mm[0]) + 0.3 * self.frame_size_cart_mm[0],
                torch.tensor(self.frame_size_cart_mm[1]),
            )  # 0.3
            self.phi_range = [torch.min(phi_1, phi_2).item(), torch.max(phi_1, phi_2).item()]
        elif angular_limits is None and angular_range is not None:
            angular_range -= 0.01
            _, phi_ref = self.get_pol_from_cart(
                torch.tensor(self.source_pos_slice_pix[0]),
                torch.tensor(self.frame_size_cart_pix[1]),
            )
            self.phi_range = [
                phi_ref.item() - angular_range / 2,
                phi_ref.item() + angular_range / 2,
            ]
        else:
            _, phis = self.get_pol_from_cart(
                torch.tensor(np.asarray(angular_limits)[:, 0]),
                torch.tensor(np.asarray(angular_limits)[:, 1]),
            )
            self.phi_range = list(phis.numpy()[::-1])

        # The estimated radii are tightened slightly to avoid sampling just
        # outside the observed fan support, where interpolation would otherwise
        # mix foreground with background zeros.
        if radius_range_mm is None:
            self.t_range_pix = np.asarray(
                [
                    (int)(self.frame_size_cart_mm[1] * 0.18 - self.source_pos_slice_mm[1]),
                    self.frame_size_cart_mm[1] - self.source_pos_slice_mm[1] - 1,
                ]
            )  # 0.7
        else:
            # sometimes the larger radius is too small by errors in source finding. This line corrects for it.
            if ((radius_range_mm + self.source_pos_slice_mm[1]) / self.frame_size_cart_mm)[1] < 0.8:
                radius_range_mm[1] = (self.frame_size_cart_mm[1] * 0.9) - self.source_pos_slice_mm[1]
            radius_range_mm[0] += 2
            radius_range_mm[1] -= 2
            self.t_range_pix = np.asarray(radius_range_mm)

        self.axis_added_dimension = axis_added_dimension

    def get_cart_from_pol(self, r: torch.Tensor, phi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert polar fan coordinates to Cartesian image coordinates.

        Args:
            r: Radii in millimetres.
            phi: Angles in radians with the same shape as ``r``.

        Returns:
            Tuple ``(x, y)`` in image pixels.
        """
        assert (
            r.shape == phi.shape
        ), f"r and phi tensors need to have same shape but have r: {r.shape}, phi: {phi.shape}"

        x = r * torch.cos(phi) / self.pixel_spacing[0]
        y = r * torch.sin(phi) / self.pixel_spacing[1]

        return (x + self.source_pos_slice_pix[0], y + self.source_pos_slice_pix[1])

    def get_pol_from_cart(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert Cartesian image coordinates to polar fan coordinates.

        Args:
            x: Row-like image coordinates in pixels.
            y: Column-like image coordinates in pixels.

        Returns:
            Tuple ``(r, phi)`` where ``r`` is in millimetres and ``phi`` is in
            radians.
        """
        assert x.shape == y.shape, f"x and y tensors need to have same shape but have x: {x.shape}, y: {y.shape}"

        # relative to sound source
        x = x - self.source_pos_slice_pix[0]
        y = y - self.source_pos_slice_pix[1]

        # from pix to mm
        x = x * self.pixel_spacing[0]
        y = y * self.pixel_spacing[1]

        # get polar coords
        r = torch.sqrt(x * x + y * y)
        phi = torch.arctan2(y, x)

        return (r, phi)

    def get_reference_cart_grid(self) -> torch.Tensor:
        """Return Cartesian coordinates for one untransformed fan slice.

        Returns:
            Tensor with shape ``(radial_samples, angular_samples, 2)``.
        """

        # Build the fan in polar coordinates first so equal scanline spacing is
        # preserved before converting to Cartesian sampling coordinates.
        phi_arr = torch.linspace(self.phi_range[0], self.phi_range[1], self.frame_size_pol[1])
        r_arr = torch.linspace(self.t_range_pix[0], self.t_range_pix[1], self.frame_size_pol[0])
        phi_tensor = torch.tile(phi_arr, (self.frame_size_pol[0], 1))
        r_tensor = torch.tile(r_arr.reshape(-1, 1), (1, self.frame_size_pol[1]))

        x_tensor, y_tensor = self.get_cart_from_pol(r_tensor, phi_tensor)

        reference_tensor = torch.cat((x_tensor[:, :, None], y_tensor[:, :, None]), dim=2)

        return reference_tensor

    def get_cart_coord_FoV(self, aff_trans_mat: torch.Tensor) -> torch.Tensor:
        """Return Cartesian volume coordinates for all requested fan slices.

        Args:
            aff_trans_mat: Affine matrices with shape ``(batch, 4, 4)``.

        Returns:
            Cartesian coordinates with shape
            ``(batch, radial_samples, angular_samples, 3)``.
        """
        bs = aff_trans_mat.shape[0]
        assert torch.Size((bs, 4, 4)) == aff_trans_mat.shape, "Shape mismatch. The input must be of shape (bs,4,4)."
        device = aff_trans_mat.device

        # Start with the 2D fan grid, insert the missing volume axis, then move
        # coordinates so rotations happen around the estimated tilt source.
        slice_coord = torch.tile(self.get_reference_cart_grid()[None, :, :, :], (bs, 1, 1, 1))
        slice_coord = torch.cat(
            [
                slice_coord[:, :, :, : self.axis_added_dimension],
                torch.zeros(slice_coord.shape[:3])[:, :, :, None],
                slice_coord[:, :, :, self.axis_added_dimension :],
            ],
            dim=3,
        ).to(device=device)

        T_center = torch.tile(
            -torch.tensor(
                np.insert(
                    np.asarray([self.source_pos_tilt_pix[0], self.source_pos_tilt_pix[1]]),
                    self.axis_added_dimension,
                    0,
                ),
                dtype=slice_coord.dtype,
            )[None, :],
            (bs, 1),
        )[:, None, None, :].to(device=device)
        slice_coord = slice_coord + T_center

        slice_coord = transform_slice(slice_coord, aff_trans_mat)

        return slice_coord


class SliceHandlerLinearProbe:
    """Generate Cartesian coordinate grids for aligned linear-probe stacks."""

    def __init__(
        self,
        frame_size_pix: list[int],  # x,y in pix
        source_pos_slice_pix: list[float],  # x,y
        pixel_spacing_cart: list = [1, 1],  # x,y
        axis_added_dimension: int = 1,
    ) -> None:
        """Construct a linear-probe slice handler.

        Args:
            frame_size_pix: Cartesian slice size in pixels as ``(height, width)``.
            source_pos_slice_pix: Slice source/origin in pixels.
            pixel_spacing_cart: Pixel spacing in millimetres.
            axis_added_dimension: Axis where the slice plane is embedded into
                3D coordinates.
        """

        self.frame_size_cart_pix = frame_size_pix  # (H,W)
        self.frame_size_cart_mm = np.asarray(self.frame_size_cart_pix) * pixel_spacing_cart
        self.source_pos_slice_pix = np.asarray(source_pos_slice_pix)  # relative to frame 0,0
        self.source_pos_slice_mm = self.source_pos_slice_pix * pixel_spacing_cart
        self.pixel_spacing = pixel_spacing_cart

        self.axis_added_dimension = axis_added_dimension

    def get_cart_from_pol(self, y: torch.Tensor, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return identity coordinates for linear-probe data.

        Args:
            y: Row coordinates.
            x: Column coordinates.

        Returns:
            Tuple ``(y, x)`` unchanged.
        """

        return (y, x)

    def get_pol_from_cart(self, y: torch.Tensor, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return identity coordinates for linear-probe data.

        Args:
            y: Row coordinates.
            x: Column coordinates.

        Returns:
            Tuple ``(y, x)`` unchanged.
        """

        return (y, x)

    def get_reference_cart_grid(self) -> torch.Tensor:
        """Return Cartesian coordinates for one untransformed linear slice.

        Returns:
            Tensor with shape ``(height, width, 3)``.
        """
        y, x = torch.meshgrid(
            torch.arange(self.frame_size_cart_pix[0]), torch.arange(self.frame_size_cart_pix[1]), indexing="ij"
        )
        reference_tensor = torch.stack([y, x, torch.zeros_like(x)], dim=-1)

        return reference_tensor

    def get_cart_coord_FoV(self, aff_trans_mat: torch.Tensor) -> torch.Tensor:
        """Return Cartesian stack coordinates for selected linear slices.

        Args:
            aff_trans_mat: Affine matrices with shape ``(batch, 4, 4)``.

        Returns:
            Cartesian coordinates with shape ``(batch, height, width, 3)``.
        """
        bs = aff_trans_mat.shape[0]
        assert torch.Size((bs, 4, 4)) == aff_trans_mat.shape, "Shape mismatch. The input must be of shape (bs,4,4)."
        device = aff_trans_mat.device

        # Linear data is already Cartesian, so the reference grid can be used
        # directly before applying the per-slice affine transform.
        slice_coord = torch.tile(self.get_reference_cart_grid()[None, :, :, :], (bs, 1, 1, 1)).to(device=device)

        slice_coord = transform_slice(slice_coord, aff_trans_mat)

        return slice_coord


def trilinear_interpolation(
    coord_array: torch.Tensor, volume_array: torch.Tensor, padding: bool = True
) -> torch.Tensor:
    """Interpolate values at 3D Cartesian coordinates with trilinear weights.

    Args:
        coord_array: Coordinates with shape ``(batch, height, width, 3)``.
        volume_array: Volume or parameter maps with shape
            ``(height, width, depth, channels)``.
        padding: Whether to add the interpolation padding expected by the
            coordinate convention.

    Returns:
        Interpolated values with shape ``(batch, height, width, channels)``.
    """

    coord_array_shape = coord_array.shape
    assert len(coord_array_shape) == 4, "Most likely batch dimension missing"
    assert len(volume_array.shape) == 4, "Most likely list dimension missing."
    volume_array = volume_array.to(device=coord_array.device)

    if padding:
        volume_array = torch.nn.functional.pad(volume_array, (0, 0, 3, 3, 3, 3, 3, 3), mode="constant", value=0)
    vol_shape = np.asarray(volume_array.shape[:3])
    vol_shape -= 6

    # Coordinates are shifted by three voxels to account for the explicit
    # padding. This lets slices near the original boundary interpolate safely.
    coords = (coord_array + 3).flatten(0, -2)
    mask_x = torch.logical_and(coords[:, 0] <= vol_shape[0] + 2, coords[:, 0] >= 0)
    mask_y = torch.logical_and(coords[:, 1] <= vol_shape[1] + 2, coords[:, 1] >= 0)
    mask_z = torch.logical_and(coords[:, 2] <= vol_shape[2] + 2, coords[:, 2] >= 0)
    mask = torch.logical_and(mask_x, torch.logical_and(mask_y, mask_z))
    coords = coords[mask, :]

    # Gather the eight neighbouring lattice points for every valid coordinate.
    x_floor = torch.floor(coords[:, 0]).int()
    y_floor = torch.floor(coords[:, 1]).int()
    z_floor = torch.floor(coords[:, 2]).int()
    x_ceil = x_floor + 1
    y_ceil = y_floor + 1
    z_ceil = z_floor + 1

    values_loc = torch.hstack(
        [
            volume_array[x_floor, y_floor, z_floor, None],
            volume_array[x_floor, y_ceil, z_ceil, None],
            volume_array[x_floor, y_floor, z_ceil, None],
            volume_array[x_floor, y_ceil, z_floor, None],
            volume_array[x_ceil, y_floor, z_floor, None],
            volume_array[x_ceil, y_ceil, z_ceil, None],
            volume_array[x_ceil, y_floor, z_ceil, None],
            volume_array[x_ceil, y_ceil, z_floor, None],
        ]
    ).swapaxes(0, 1)

    # The signed terms match the original interpolation convention.
    weights = torch.vstack(
        [
            (x_ceil - coords[:, 0]) * (y_ceil - coords[:, 1]) * (z_ceil - coords[:, 2]),
            (coords[:, 0] - x_ceil) * (coords[:, 1] - y_floor) * (z_floor - coords[:, 2]),
            (coords[:, 0] - x_ceil) * (y_ceil - coords[:, 1]) * (z_floor - coords[:, 2]),
            (x_ceil - coords[:, 0]) * (coords[:, 1] - y_floor) * (z_ceil - coords[:, 2]),
            (x_floor - coords[:, 0]) * (y_ceil - coords[:, 1]) * (coords[:, 2] - z_ceil),
            (coords[:, 0] - x_floor) * (coords[:, 1] - y_floor) * (coords[:, 2] - z_floor),
            (coords[:, 0] - x_floor) * (y_ceil - coords[:, 1]) * (coords[:, 2] - z_floor),
            (x_floor - coords[:, 0]) * (coords[:, 1] - y_floor) * (coords[:, 2] - z_ceil),
        ]
    )

    weights = torch.tile(weights[:, :, None], (1, 1, values_loc.shape[-1]))
    values_loc = torch.sum(weights * values_loc, dim=0)
    values = torch.zeros((mask.shape[0], values_loc.shape[-1]), dtype=torch.float32).to(device=coord_array.device)
    values[mask] = values_loc
    values = values.reshape((*tuple(coord_array_shape[:-1]), values_loc.shape[-1]))

    return values


def pseudo_inverse_trilinear_interpolation(
    coord_array: torch.Tensor, volume_shape: tuple, values: torch.Tensor
) -> torch.Tensor:
    """Scatter slice values back into a 3D volume with trilinear weights.

    Args:
        coord_array: Coordinates with shape ``(batch, height, width, 3)``.
        volume_shape: Unpadded output volume shape.
        values: Values corresponding to ``coord_array`` with shape
            ``(batch, height, width)``.

    Returns:
        Reconstructed volume with shape ``volume_shape``.

    Raises:
        AssertionError: If ``values`` and ``coord_array`` shapes do not match.
    """

    assert values.shape == coord_array.shape[:-1], "Mismatch between values and coords shape"

    device = coord_array.device

    coords = (coord_array + 3).flatten(0, -2)
    values = values.flatten()
    mask_x = torch.logical_and(coords[:, 0] <= volume_shape[0] + 2, coords[:, 0] >= 0)
    mask_y = torch.logical_and(coords[:, 1] <= volume_shape[1] + 2, coords[:, 1] >= 0)
    mask_z = torch.logical_and(coords[:, 2] <= volume_shape[2] + 2, coords[:, 2] >= 0)
    mask = torch.logical_and(mask_x, torch.logical_and(mask_y, mask_z))
    coords = coords[mask, :]
    values = values[mask]

    # Distribute each sampled value to the eight voxels that would contribute to
    # it during forward trilinear interpolation.
    x_floor = torch.floor(coords[:, 0]).int()
    y_floor = torch.floor(coords[:, 1]).int()
    z_floor = torch.floor(coords[:, 2]).int()
    x_ceil = x_floor + 1
    y_ceil = y_floor + 1
    z_ceil = z_floor + 1

    # Get nearest surrounding integer values for each point.
    dep = torch.cat(
        [
            torch.vstack([x_floor, y_floor, z_floor])[..., None],
            torch.vstack([x_floor, y_ceil, z_ceil])[..., None],
            torch.vstack([x_floor, y_floor, z_ceil])[..., None],
            torch.vstack([x_floor, y_ceil, z_floor])[..., None],
            torch.vstack([x_ceil, y_floor, z_floor])[..., None],
            torch.vstack([x_ceil, y_ceil, z_ceil])[..., None],
            torch.vstack([x_ceil, y_floor, z_ceil])[..., None],
            torch.vstack([x_ceil, y_ceil, z_floor])[..., None],
        ],
        dim=2,
    ).reshape((3, values.shape[0] * 8))

    # Get the weighting of inverse interpolation.
    weights = torch.vstack(
        [
            (x_ceil - coords[:, 0]) * (y_ceil - coords[:, 1]) * (z_ceil - coords[:, 2]),
            (coords[:, 0] - x_ceil) * (coords[:, 1] - y_floor) * (z_floor - coords[:, 2]),
            (coords[:, 0] - x_ceil) * (y_ceil - coords[:, 1]) * (z_floor - coords[:, 2]),
            (x_ceil - coords[:, 0]) * (coords[:, 1] - y_floor) * (z_ceil - coords[:, 2]),
            (x_floor - coords[:, 0]) * (y_ceil - coords[:, 1]) * (coords[:, 2] - z_ceil),
            (coords[:, 0] - x_floor) * (coords[:, 1] - y_floor) * (coords[:, 2] - z_floor),
            (coords[:, 0] - x_floor) * (y_ceil - coords[:, 1]) * (coords[:, 2] - z_floor),
            (x_floor - coords[:, 0]) * (coords[:, 1] - y_floor) * (coords[:, 2] - z_ceil),
        ]
    ).reshape(values.shape[0] * 8)

    values = torch.cat([values[:, None] for _ in range(8)], dim=1).flatten()

    volume_values = torch.zeros(tuple(np.asarray(volume_shape) + 6)).to(device=device)

    # ``torch.unique`` on MPS is routed through CPU for compatibility with
    # current backend support.
    if device.type == "mps":
        unique_points, idxs = torch.unique(dep.to(device="cpu"), dim=1, return_inverse=True)
        unique_points = unique_points.to(device=device)
        idxs = idxs.to(device=device)
    else:
        unique_points, idxs = torch.unique(dep, dim=1, return_inverse=True)

    unique_values = torch.zeros(unique_points.shape[1]).to(device=device)
    unique_weights = torch.full_like(unique_values, 1e-9).to(device=device)

    unique_values.index_add_(0, idxs, values * weights)
    unique_weights.index_add_(0, idxs, weights)

    volume_values[tuple(unique_points)] = unique_values / unique_weights

    return volume_values[3:-3, 3:-3, 3:-3]


def pseudo_inverse_bilinear_interpolation(
    coord_array: torch.Tensor, imgstack_shape: tuple | np.ndarray, values: torch.Tensor
) -> torch.Tensor:
    """Scatter slice values back into an image stack with bilinear weights.

    Args:
        coord_array: Coordinates with shape ``(batch, height, width, 3)``.
        imgstack_shape: Unpadded output image-stack shape.
        values: Values corresponding to ``coord_array`` with shape
            ``(batch, height, width)``.

    Returns:
        Reconstructed image stack with shape ``imgstack_shape``.

    Raises:
        AssertionError: If ``values`` and ``coord_array`` shapes do not match.
    """

    assert values.shape == coord_array.shape[:-1], "Mismatch between values and coords shape"
    coord_array = coord_array.clone()

    device = coord_array.device

    coords = coord_array.flatten(0, -2)
    coords[:, :2] += 3
    values = values.flatten()
    mask_x = torch.logical_and(coords[:, 0] <= imgstack_shape[0] + 2, coords[:, 0] >= 0)
    mask_y = torch.logical_and(coords[:, 1] <= imgstack_shape[1] + 2, coords[:, 1] >= 0)
    mask = torch.logical_and(mask_x, mask_y)
    coords = coords[mask, :]
    values = values[mask]

    # Bilinear inverse interpolation operates within each stack slice; z is an
    # integer slice index rather than an interpolated coordinate.
    x_floor = torch.floor(coords[:, 0]).int()
    y_floor = torch.floor(coords[:, 1]).int()
    z_int = coords[:, 2].int()
    x_ceil = x_floor + 1
    y_ceil = y_floor + 1

    dep = torch.cat(
        [
            torch.vstack([x_floor, y_floor, z_int])[..., None],
            torch.vstack([x_floor, y_ceil, z_int])[..., None],
            torch.vstack([x_ceil, y_floor, z_int])[..., None],
            torch.vstack([x_ceil, y_ceil, z_int])[..., None],
        ],
        dim=2,
    ).reshape((3, values.shape[0] * 4))

    weights = torch.vstack(
        [
            (x_ceil - coords[:, 0]) * (y_ceil - coords[:, 1]),
            (x_ceil - coords[:, 0]) * (coords[:, 1] - y_floor),
            (coords[:, 0] - x_floor) * (y_ceil - coords[:, 1]),
            (coords[:, 0] - x_floor) * (coords[:, 1] - y_floor),
        ]
    ).reshape(values.shape[0] * 4)

    values = torch.cat([values[:, None] for _ in range(4)], dim=1).flatten()

    imgstack_shape = np.asarray(imgstack_shape)
    imgstack_shape[:-1] += 6
    stack_values = torch.zeros(tuple(imgstack_shape)).to(device=device)

    # ``torch.unique`` on MPS is routed through CPU for compatibility with
    # current backend support.
    if device.type == "mps":
        unique_points, idxs = torch.unique(dep.to(device="cpu"), dim=1, return_inverse=True)
        unique_points = unique_points.to(device=device)
        idxs = idxs.to(device=device)
    else:
        unique_points, idxs = torch.unique(dep, dim=1, return_inverse=True)

    unique_values = torch.zeros(unique_points.shape[1]).to(device=device)
    unique_weights = torch.full_like(unique_values, 1e-9).to(device=device)

    unique_values.index_add_(0, idxs, values * weights)
    unique_weights.index_add_(0, idxs, weights)

    stack_values[tuple(unique_points)] = unique_values / unique_weights

    return stack_values[3:-3, 3:-3]


def bilinear_interpolation_stack(coord_array: torch.Tensor, stack: torch.Tensor, padding: bool = True) -> torch.Tensor:
    """Interpolate values from an aligned image stack with bilinear weights.

    Args:
        coord_array: Coordinates with shape ``(batch, height, width, 3)``.
        stack: Image stack or parameter maps with shape
            ``(height, width, num_slices, channels)``.
        padding: Whether to add the interpolation padding expected by the
            coordinate convention.

    Returns:
        Interpolated values with shape ``(batch, height, width, channels)``.
    """

    coord_array_shape = coord_array.shape
    assert len(coord_array_shape) == 4, "Most likely batch dimension missing"
    assert len(stack.shape) == 4, "Most likely list dimension missing."
    stack_shape = torch.tensor(stack.shape[:3])

    if padding:
        stack = torch.nn.functional.pad(stack, (0, 0, 0, 0, 3, 3, 3, 3), mode="constant", value=0)
    else:
        stack_shape[0] = stack_shape[0] - 6
        stack_shape[1] = stack_shape[1] - 6

    coords = coord_array.flatten(0, -2)
    coords[:, :2] += 3
    mask_x = torch.logical_and(coords[:, 0] <= stack_shape[0] + 2, coords[:, 0] >= 0)
    mask_y = torch.logical_and(coords[:, 1] <= stack_shape[1] + 2, coords[:, 1] >= 0)
    mask = torch.logical_and(mask_x, mask_y)
    coords = coords[mask, :]

    # Gather the four neighbouring pixels within each integer-indexed slice.
    x_floor = torch.floor(coords[:, 0]).int()
    y_floor = torch.floor(coords[:, 1]).int()
    z_int = coords[:, 2].int()
    x_ceil = x_floor + 1
    y_ceil = y_floor + 1

    values_loc = torch.hstack(
        [
            stack[x_floor, y_floor, z_int, None],
            stack[x_floor, y_ceil, z_int, None],
            stack[x_ceil, y_floor, z_int, None],
            stack[x_ceil, y_ceil, z_int, None],
        ]
    ).swapaxes(0, 1)

    # The signed terms match the original interpolation convention.
    weights = torch.vstack(
        [
            (x_ceil - coords[:, 0]) * (y_ceil - coords[:, 1]),
            (x_ceil - coords[:, 0]) * (coords[:, 1] - y_floor),
            (coords[:, 0] - x_floor) * (y_ceil - coords[:, 1]),
            (coords[:, 0] - x_floor) * (coords[:, 1] - y_floor),
        ]
    )

    weights = torch.tile(weights[:, :, None], (1, 1, values_loc.shape[-1]))
    values_loc = torch.sum(weights * values_loc, dim=0)
    values = torch.zeros((mask.shape[0], values_loc.shape[-1]), dtype=torch.float32).to(device=coord_array.device)
    values[mask] = values_loc
    values = values.reshape((*tuple(coord_array_shape[:-1]), values_loc.shape[-1]))

    return values
