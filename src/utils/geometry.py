###### imports ######

import numpy as np
import torch
import sys
from src.utils.transformations import transform_slice
from src.utils.visualization import visualize_vol

###### body ######


class VolumeHandler:
    def __init__(
        self,
        volume: torch.Tensor,
        pixel_spacing: list[float] | np.ndarray = [1, 1, 1],
        volume_mask: torch.Tensor | None = None,
    ):
        """Constructor of Volume Handler. The volume handler is mainly designe to hold volume and corresponding mask
        as well as interpolate.

        Args:
            volume (torch.Tensor): Volume to create volume handler of
            pixel_spacing (list[float], optional): Pixcel spacing of pixels in volume. Defaults to np.asarray([1,1,1]).
            volume_mask (torch.Tensor | None): Volume mask. Defaults to None

        Example:
            >>> volume = vol = np.random.randint(0,255,(60,60,60))
            >>> vh = VolumeHandler(volume)
        """

        assert volume.ndim == 3, "Volume tensor must be 3D"
        assert volume.ndim == pixel_spacing.__len__(), "Pixcel spacing must have size 3 but is of "

        # masking cone from background
        if volume_mask is None:
            volume_mask = volume.clone()
            volume_mask[volume_mask > 0] = 1
            volume_mask[volume_mask < 0] = 0
        self.vol_mask = volume_mask

        self.shape = volume.shape

        self.volume = volume
        self.pixel_spacing = pixel_spacing

    def get_pixel_value_cart(self, coord_array: torch.Tensor) -> torch.Tensor:
        """Returns the grey values of a set of slices defined by cartesian coordinates.

        Args:
            coord_array (torch.Tensor): Array of coordinates. Shape: (batch, r_range, phi_range, 3)

        Returns:
            torch.Tensor: Tensor containing grey values. Shape: (batch, r_range, phi_range)
        """
        return trilinear_interpolation(coord_array, self.volume[..., None])[..., -1]

    def get_mask_value_cart(self, coord_array: torch.Tensor) -> torch.Tensor:
        """Returns the values of the volume mask of a set of slices defined by cartesian coordinates.

        Args:
            coord_array (torch.Tensor): Array of coordinates. Shape: (batch, r_range, phi_range, 3)

        Returns:
            torch.Tensor: Tensor containing grey values. Shape: (batch, r_range, phi_range)
        """
        interpolation = trilinear_interpolation(coord_array, self.vol_mask[..., None])[..., -1]
        return interpolation > 0

    def visualize_volume(self) -> None:
        """Helper function. Visualizes the volume."""
        visualize_vol(self.volume, self.pixel_spacing)

    def visualize_mask(self) -> None:
        """Visualizes the Volume Mask."""
        visualize_vol(self.vol_mask, self.pixel_spacing)


class StackHandler:
    def __init__(
        self,
        stack: torch.Tensor,
        pixel_spacing: list[float] = [1, 1, 1],
        volume_mask: torch.Tensor | None = None,
    ):
        """Constructor of Volume Handler. The volume handler is mainly designe to hold volume and corresponding mask as well as interpolate.

        Args:
            volume (torch.Tensor): Volume to create volume handler of
            pixel_spacing (list[float], optional): Pixcel spacing of pixels in volume. Defaults to np.asarray([1,1,1]).
            volume_mask (torch.Tensor | None): Volume mask. Defaults to None

        Example:
            >>> volume = vol = np.random.randint(0,255,(60,60,60))
            >>> vh = VolumeHandler(volume)
        """

        assert stack.ndim == 3, "Volume tensor must be 3D"
        assert stack.ndim == pixel_spacing.__len__(), "Pixcel spacing must have size 3 but is of "

        # masking cone from background
        if volume_mask is None:
            volume_mask = stack.clone()
            volume_mask[volume_mask > 0] = 1
            volume_mask[volume_mask < 0] = 0
        self.vol_mask = volume_mask

        self.shape = stack.shape

        self.volume = stack
        self.pixel_spacing = pixel_spacing

    def get_pixel_value_cart(self, coord_array: torch.Tensor) -> torch.Tensor:
        """Returns the grey values of a set of slices defined by cartesian coordinates.

        Args:
            coord_array (torch.Tensor): Array of coordinates. Shape: (batch, r_range, phi_range, 3)

        Returns:
            torch.Tensor: Tensor containing grey values. Shape: (batch, r_range, phi_range)
        """
        return bilinear_interpolation_stack(coord_array, self.volume[..., None])[..., -1]

    def get_mask_value_cart(self, coord_array: torch.Tensor) -> torch.Tensor:
        """Returns the values of the volume mask of a set of slices defined by cartesian coordinates.

        Args:
            coord_array (torch.Tensor): Array of coordinates. Shape: (batch, r_range, phi_range, 3)

        Returns:
            torch.Tensor: Tensor containing grey values. Shape: (batch, r_range, phi_range)
        """
        interpolation = bilinear_interpolation_stack(coord_array, self.vol_mask[..., None])[..., -1]
        return interpolation > 0

    def visualize_stack(self) -> None:
        """Helper function. Visualizes the volume."""
        visualize_vol(self.volume, self.pixel_spacing)

    def visualize_mask(self) -> None:
        """Visualizes the Volume Mask."""
        visualize_vol(self.vol_mask, self.pixel_spacing)


class SliceHandler:
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
        """Constructor for SliceHandler. The SliceHandler interpolates single slices and presents them.

        Args:
            frame_size_cart (list[int]): Dimenstions of frame of slice in pixel. (x,y)
            frame_size_polar (list[int]): Discretization of the fan shaped slice in polar coordinates (r, phi)
            source_pos_slice (list[float]): Position of the sound source of the fan shaped area of interst of the slice
            source_pos_tilt (list[float] | None, optional): Position of the point around which the internal scanner head turned. Needed for some interpolation. Defaults to None.
            pixel_spacing (list[float], optional): Pixel spacing of slice in mm. Defaults to np.asarray([1,1]).
            radius_range (list | None, optional): Range of radii defining the fan shaped area of interst.
            angular_range (float | None, optional): Angle width defining the fan shaped area of interst.
            angular_limits (list | None, optional): Angular limit points. Defaults to None.
            axis_added_dimension (int): Axis along with the dimention to 3D will be added. Defaults to 1

        Example:
            >>> #TODO
        """

        assert frame_size_polar is not None, "frame_size_polar must be specified."

        # The cone is usually spanned by a curved 1D piezo crystal array, which is tilted inside the 3D scanner head.
        # source pos slice denotes the points source of the curved array while tilt referes to the point around which
        # the scanner is turned.
        if source_pos_tilt_pix is None:
            self.source_pos_tilt_pix = np.asarray(source_pos_slice_pix)
            print(
                f"not specifying source_pos_pivot is depricated. please specity explicitly",
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

        # angular range
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

        # radial range
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

        # adding dimension
        self.axis_added_dimension = axis_added_dimension

    def get_cart_from_pol(self, r: torch.Tensor, phi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Converts polar coordinates to cartesian coordinates

        Args:
            r (torch.Tensor): Tensor containing the radii of the polar coordinates
            phi (torch.Tensor): Tensor of the same shape as r containing the angular information

        Returns:
            tuple[torch.Tensor, torch.Tensor]: tuple of (x, y)

        Example:
            >>> #TODO
        """
        assert (
            r.shape == phi.shape
        ), f"r and phi tensors need to have same shape but have r: {r.shape}, phi: {phi.shape}"

        x = r * torch.cos(phi) / self.pixel_spacing[0]
        y = r * torch.sin(phi) / self.pixel_spacing[1]

        return (x + self.source_pos_slice_pix[0], y + self.source_pos_slice_pix[1])

    def get_pol_from_cart(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """converst cartesian coordinates to polar coordinates.

        Args:
            x (torch.Tensor): tensor containing x values in pixel
            y (torch.Tensor): tensor containing y values in pixel

        Returns:
            tuple[torch.Tensor, torch.Tensor]: (r,phi) in mm

        Example:
            >>> #TODO
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
        """Returns a tensor containing x, y coordinates of a prototype slice.

        Returns:
            torch.Tensor: Tensor of shape (r_range, phi_range, 2)

        Example:
            >>> #TODO
        """

        # create tensors for r and phi of fan in polar coords
        phi_arr = torch.linspace(self.phi_range[0], self.phi_range[1], self.frame_size_pol[1])
        r_arr = torch.linspace(self.t_range_pix[0], self.t_range_pix[1], self.frame_size_pol[0])
        phi_tensor = torch.tile(phi_arr, (self.frame_size_pol[0], 1))
        r_tensor = torch.tile(r_arr.reshape(-1, 1), (1, self.frame_size_pol[1]))

        # convert from polar to cart
        x_tensor, y_tensor = self.get_cart_from_pol(r_tensor, phi_tensor)

        # combine
        reference_tensor = torch.cat((x_tensor[:, :, None], y_tensor[:, :, None]), dim=2)

        return reference_tensor

    def get_cart_coord_FoV(self, aff_trans_mat: torch.Tensor) -> torch.Tensor:
        """Returns tensor containing cartesian coordinates of the fans of al slices in cartesian coordinates.

        Args:
            aff_trans_mat (torch.Tensor): Tensor containg affine transformation matrices for each slice.

        Returns:
            torch.Tensor: Cartesian coordinates of all slices. Shape (batch, r_range, phi_range, 3)
        """
        bs = aff_trans_mat.shape[0]
        assert torch.Size((bs, 4, 4)) == aff_trans_mat.shape, "Shaoe mismatch. The input must be of shape (bs,4,4)."
        device = aff_trans_mat.device

        # get slice coords in pix
        slice_coord = torch.tile(self.get_reference_cart_grid()[None, :, :, :], (bs, 1, 1, 1))
        slice_coord = torch.cat(
            [
                slice_coord[:, :, :, : self.axis_added_dimension],
                torch.zeros(slice_coord.shape[:3])[:, :, :, None],
                slice_coord[:, :, :, self.axis_added_dimension :],
            ],
            dim=3,
        ).to(device=device)

        # compensate for mismatch of slice source and volume origin
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

        # transform coordinates
        slice_coord = transform_slice(slice_coord, aff_trans_mat)

        return slice_coord


class SliceHandlerLinearProbe:
    def __init__(
        self,
        frame_size_pix: list[int],  # x,y in pix
        source_pos_slice_pix: list[float],  # x,y
        pixel_spacing_cart: list = [1, 1],  # x,y
        axis_added_dimension: int = 1,
    ) -> None:
        """Constructor for SliceHandler. The SliceHandler interpolates single slices and presents them.

        Args:
            frame_size_cart (list[int]): Dimenstions of frame of slice in pixel. (x,y)
            frame_size_polar (list[int]): Discretization of the fan shaped slice in polar coordinates (r, phi)
            source_pos_slice (list[float]): Position of the sound source of the fan shaped area of interst of the slice
            source_pos_tilt (list[float] | None, optional): Position of the point around which the internal scanner head turned. Needed for some interpolation. Defaults to None.
            pixel_spacing (list[float], optional): Pixel spacing of slice in mm. Defaults to np.asarray([1,1]).
            radius_range (list | None, optional): Range of radii defining the fan shaped area of interst. Defaults to None.
            angular_range (float | None, optional): Angle width defining the fan shaped area of interst. Defaults to None.
            angular_limits (list | None, optional): Angular limit points. Defaults to None.
            axis_added_dimension (int): Axis along with the dimention to 3D will be added. Defaults to 1

        Example:
            >>> #TODO
        """

        self.frame_size_cart_pix = frame_size_pix  # (H,W)
        self.frame_size_cart_mm = np.asarray(self.frame_size_cart_pix) * pixel_spacing_cart
        self.source_pos_slice_pix = np.asarray(source_pos_slice_pix)  # relative to frame 0,0
        self.source_pos_slice_mm = self.source_pos_slice_pix * pixel_spacing_cart
        self.pixel_spacing = pixel_spacing_cart

        # adding dimension
        self.axis_added_dimension = axis_added_dimension

    def get_cart_from_pol(self, y: torch.Tensor, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return (y, x)

    def get_pol_from_cart(self, y: torch.Tensor, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return (y, x)

    def get_reference_cart_grid(self) -> torch.Tensor:
        """Returns a tensor containing x, y coordinates of a prototype slice.

        Returns:
            torch.Tensor: Tensor of shape (r_range, phi_range, 2)

        Example:
            >>> #TODO
        """
        y, x = torch.meshgrid(
            torch.arange(self.frame_size_cart_pix[0]), torch.arange(self.frame_size_cart_pix[1]), indexing="ij"
        )
        reference_tensor = torch.stack([y, x, torch.zeros_like(x)], dim=-1)

        return reference_tensor

    def get_cart_coord_FoV(self, aff_trans_mat: torch.Tensor) -> torch.Tensor:
        """Returns tensor containing cartesian coordinates of the fans of al slices in cartesian coordinates.

        Args:
            aff_trans_mat (torch.Tensor): Tensor containg affine transformation matrices for each slice. Shape: (batch, 4, 4)

        Returns:
            torch.Tensor: Cartesian coordinates of all slices. Shape (batch, r_range, phi_range, 3)
        """
        bs = aff_trans_mat.shape[0]
        assert torch.Size((bs, 4, 4)) == aff_trans_mat.shape, f"Shaoe mismatch. The input must be of shape (bs,4,4)."
        device = aff_trans_mat.device

        # get slice coords in pix
        slice_coord = torch.tile(self.get_reference_cart_grid()[None, :, :, :], (bs, 1, 1, 1)).to(device=device)

        # transform coordinates
        slice_coord = transform_slice(slice_coord, aff_trans_mat)

        return slice_coord


def trilinear_interpolation(
    coord_array: torch.Tensor, volume_array: torch.Tensor, padding: bool = True
) -> torch.Tensor:
    """Returns values for the 3D coordinates of coord_array, interpolated from volume using trilinear interpolation

    Args:
        coord_array (torch.Tensor): Tensor of shape (batch,...,3) containing 3D carthesian coordinates
        volume_array (torch.Tensor): Array of n volumes to interpolate from. Dimensions: (H,W,D,n).
        padding (bool): Specifies if padding of volume is required. Defaults to True

    Returns:
        torch.Tensor: Interpolated values
    """

    coord_array_shape = coord_array.shape
    assert len(coord_array_shape) == 4, f"Most likely batch dimension missing"
    assert len(volume_array.shape) == 4, "Most likely list dimension missing."
    volume_array = volume_array.to(device=coord_array.device)

    if padding:
        volume_array = torch.nn.functional.pad(volume_array, (0, 0, 3, 3, 3, 3, 3, 3), mode="constant", value=0)
    vol_shape = np.asarray(volume_array.shape[:3])
    vol_shape -= 6

    coords = (coord_array + 3).flatten(0, -2)
    mask_x = torch.logical_and(coords[:, 0] <= vol_shape[0] + 2, coords[:, 0] >= 0)
    mask_y = torch.logical_and(coords[:, 1] <= vol_shape[1] + 2, coords[:, 1] >= 0)
    mask_z = torch.logical_and(coords[:, 2] <= vol_shape[2] + 2, coords[:, 2] >= 0)
    mask = torch.logical_and(mask_x, torch.logical_and(mask_y, mask_z))
    coords = coords[mask, :]

    # get surounding points
    x_floor = torch.floor(coords[:, 0]).int()
    y_floor = torch.floor(coords[:, 1]).int()
    z_floor = torch.floor(coords[:, 2]).int()
    x_ceil = x_floor + 1
    y_ceil = y_floor + 1
    z_ceil = z_floor + 1

    # get values
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

    # get corsponding weights
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

    # do interpolation
    weights = torch.tile(weights[:, :, None], (1, 1, values_loc.shape[-1]))
    values_loc = torch.sum(weights * values_loc, dim=0)
    values = torch.zeros((mask.shape[0], values_loc.shape[-1]), dtype=torch.float32).to(device=coord_array.device)
    values[mask] = values_loc
    values = values.reshape((*tuple(coord_array_shape[:-1]), values_loc.shape[-1]))

    return values


def pseudo_inverse_trilinear_interpolation(
    coord_array: torch.Tensor, volume_shape: tuple, values: torch.Tensor
) -> torch.Tensor:
    """Performes pseudo inverse interpolation. Using the points stored in coord_array and the coresponding values
    stored in values it populates a volume of shape volume_shape with inverse interpolated values.

    Args:
        coord_array (torch.Tensor): 3D cartesian coordinates
        volume_shape (tuple): Shape of output volume
        values (torch.Tensor): Values corresponding to coord_array

    Returns:
        torch.Tensor: Pseudo inverse interpolation
    """

    assert values.shape == coord_array.shape[:-1], "Missmatch between values and coords shape"

    device = coord_array.device

    coords = (coord_array + 3).flatten(0, -2)
    values = values.flatten()
    mask_x = torch.logical_and(coords[:, 0] <= volume_shape[0] + 2, coords[:, 0] >= 0)
    mask_y = torch.logical_and(coords[:, 1] <= volume_shape[1] + 2, coords[:, 1] >= 0)
    mask_z = torch.logical_and(coords[:, 2] <= volume_shape[2] + 2, coords[:, 2] >= 0)
    mask = torch.logical_and(mask_x, torch.logical_and(mask_y, mask_z))
    coords = coords[mask, :]
    values = values[mask]

    # get surounding points
    x_floor = torch.floor(coords[:, 0]).int()
    y_floor = torch.floor(coords[:, 1]).int()
    z_floor = torch.floor(coords[:, 2]).int()
    x_ceil = x_floor + 1
    y_ceil = y_floor + 1
    z_ceil = z_floor + 1

    # get nearest sourounding integer values for each point
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

    # get the weighting of inverse interpolation
    weigths = torch.vstack(
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
    ).reshape((values.shape[0] * 8))

    # multipy values
    values = torch.cat([values[:, None] for i in range(8)], dim=1).flatten()

    # empty volume
    volume_values = torch.zeros(tuple(np.asarray(volume_shape) + 6)).to(device=device)

    # find points in volume influenced by more than one point of point_array
    if device.type == "mps":
        unique_points, idxs = torch.unique(dep.to(device="cpu"), dim=1, return_inverse=True)
        unique_points = unique_points.to(device=device)
        idxs = idxs.to(device=device)
    else:
        unique_points, idxs = torch.unique(dep, dim=1, return_inverse=True)

    # for each point in the volume add its values and weights
    unique_values = torch.zeros(unique_points.shape[1]).to(device=device)
    unique_weights = torch.full_like(unique_values, 1e-9).to(device=device)

    unique_values.index_add_(0, idxs, values * weigths)
    unique_weights.index_add_(0, idxs, weigths)

    # write to volume
    volume_values[tuple(unique_points)] = unique_values / unique_weights

    return volume_values[3:-3, 3:-3, 3:-3]


def pseudo_inverse_bilinear_interpolation(
    coord_array: torch.Tensor, imgstack_shape: tuple | np.ndarray, values: torch.Tensor
) -> torch.Tensor:
    """Performes pseudo inverse interpolation. Using the points stored in coord_array and the coresponding values
    stored in values it populates a volume of shape volume_shape with inverse interpolated values.

    Args:
        coord_array (torch.Tensor): 3D cartesian coordinates
        imgstack_shape (tuple | np.ndarray): Shape of output image stack
        values (torch.Tensor): Values corresponding to coord_array

    Returns:
        torch.Tensor: Pseudo inverse interpolation
    """

    assert values.shape == coord_array.shape[:-1], "Missmatch between values and coords shape"
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

    # get surounding points
    x_floor = torch.floor(coords[:, 0]).int()
    y_floor = torch.floor(coords[:, 1]).int()
    z_int = coords[:, 2].int()
    x_ceil = x_floor + 1
    y_ceil = y_floor + 1

    # get nearest sourounding integer values for each point
    dep = torch.cat(
        [
            torch.vstack([x_floor, y_floor, z_int])[..., None],
            torch.vstack([x_floor, y_ceil, z_int])[..., None],
            torch.vstack([x_ceil, y_floor, z_int])[..., None],
            torch.vstack([x_ceil, y_ceil, z_int])[..., None],
        ],
        dim=2,
    ).reshape((3, values.shape[0] * 4))

    # get the weighting of inverse interpolation
    weigths = torch.vstack(
        [
            (x_ceil - coords[:, 0]) * (y_ceil - coords[:, 1]),
            (x_ceil - coords[:, 0]) * (coords[:, 1] - y_floor),
            (coords[:, 0] - x_floor) * (y_ceil - coords[:, 1]),
            (coords[:, 0] - x_floor) * (coords[:, 1] - y_floor),
        ]
    ).reshape((values.shape[0] * 4))

    # multipy values
    values = torch.cat([values[:, None] for i in range(4)], dim=1).flatten()

    # empty volume
    imgstack_shape = np.asarray(imgstack_shape)
    imgstack_shape[:-1] += 6
    stack_values = torch.zeros(tuple(imgstack_shape)).to(device=device)

    # find points in volume influenced by more than one point of point_array
    if device.type == "mps":
        unique_points, idxs = torch.unique(dep.to(device="cpu"), dim=1, return_inverse=True)
        unique_points = unique_points.to(device=device)
        idxs = idxs.to(device=device)
    else:
        unique_points, idxs = torch.unique(dep, dim=1, return_inverse=True)

    # for each point in the volume add its values and weights
    unique_values = torch.zeros(unique_points.shape[1]).to(device=device)
    unique_weights = torch.full_like(unique_values, 1e-9).to(device=device)

    unique_values.index_add_(0, idxs, values * weigths)
    unique_weights.index_add_(0, idxs, weigths)

    # write to volume
    stack_values[tuple(unique_points)] = unique_values / unique_weights

    return stack_values[3:-3, 3:-3]


def bilinear_interpolation_stack(coord_array: torch.Tensor, stack: torch.Tensor, padding: bool = True) -> torch.Tensor:
    """Returns values for the 3D coordinates of coord_array, interpolated from volume using trilinear interpolation

    Args:
        coord_array (torch.Tensor): Tensor of shape (batch,...,3) containing 3D carthesian coordinates
        volume_array (torch.Tensor): Array of n volumes to interpolate from. Dimensions: (b,H,W,n).
        padding (bool): Specifies if padding of volume is required. Defaults to True

    Returns:
        torch.Tensor: Interpolated values
    """

    coord_array_shape = coord_array.shape
    assert len(coord_array_shape) == 4, f"Most likely batch dimension missing"
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

    # get surounding points
    x_floor = torch.floor(coords[:, 0]).int()
    y_floor = torch.floor(coords[:, 1]).int()
    z_int = coords[:, 2].int()
    x_ceil = x_floor + 1
    y_ceil = y_floor + 1

    # get values
    values_loc = torch.hstack(
        [
            stack[x_floor, y_floor, z_int, None],
            stack[x_floor, y_ceil, z_int, None],
            stack[x_ceil, y_floor, z_int, None],
            stack[x_ceil, y_ceil, z_int, None],
        ]
    ).swapaxes(0, 1)

    # get corsponding weights
    weights = torch.vstack(
        [
            (x_ceil - coords[:, 0]) * (y_ceil - coords[:, 1]),
            (x_ceil - coords[:, 0]) * (coords[:, 1] - y_floor),
            (coords[:, 0] - x_floor) * (y_ceil - coords[:, 1]),
            (coords[:, 0] - x_floor) * (coords[:, 1] - y_floor),
        ]
    )

    # do interpolation
    weights = torch.tile(weights[:, :, None], (1, 1, values_loc.shape[-1]))
    values_loc = torch.sum(weights * values_loc, dim=0)
    values = torch.zeros((mask.shape[0], values_loc.shape[-1]), dtype=torch.float32).to(device=coord_array.device)
    values[mask] = values_loc
    values = values.reshape((*tuple(coord_array_shape[:-1]), values_loc.shape[-1]))

    return values
