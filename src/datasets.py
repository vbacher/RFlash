import torch
from torch.utils import data
import numpy as np

from src.utils.geometry import VolumeHandler, SliceHandler, StackHandler, SliceHandlerLinearProbe
from src.utils.transformations import rotmat_from_euler
from src._datatypes import VolumeTransducerGeometry
from src.trainings_params import Parameter_Demo3D


class Dataset_3D_volume(data.Dataset):
    def __init__(
        self,
        volume: np.ndarray,
        model_params: Parameter_Demo3D,
        geometry: VolumeTransducerGeometry,
        pixel_spacing_mm: list[float] | np.ndarray = [1.0, 1.0, 1.0],
        pre_rot: list[float] = [0, 0, 0],
        device: torch.device | None = None,
    ) -> None:
        """Creates pytorch dataset from a single volume, by sampling evenly spaced slices from the volume.

        Args:
            volume (np.ndarray): Volume to create dataset from.
            model_params (Parameter_Demo3D): Model parameters object
            geometry (VolumeTransducerGeometry): Volume transducer geometry object
            pixel_spacing_mm (list[float], optional): Pixel spacing in mm. Defaults to [1.,1.,1.].
            pre_rot (list[float], optional): Preliminary rotation. Defaults to [0,0,0].
            device (torch.device | None, optional): Device to use. Defaults to None.
        """

        super().__init__()
        if device is None:
            device = torch.device("cpu")

        # create volume handler
        self.vh = VolumeHandler(volume=torch.tensor(volume.copy()), pixel_spacing=pixel_spacing_mm)

        # define geometry
        self.sh = SliceHandler(
            frame_size_cart_pix=np.array(volume.shape[::2]),
            frame_size_polar=model_params.image_size_polar,
            source_pos_slice_pix=geometry.fan_source_pix[::2],
            source_pos_tilt_pix=geometry.tilt_source_pix[::2],
            pixel_spacing_cart=pixel_spacing_mm[::2],
            radius_range_mm=geometry.fan_r_range_mm,
            angular_range=geometry.fan_ang_range_rad,
        )
        assert model_params.num_fan_slices is not None, "Number of fan slices must be specified in model parameters."
        self.n = int(model_params.num_fan_slices)

        # get source shifts
        self.origin_tilt = torch.tensor(geometry.tilt_source_pix)
        self.Ts = torch.tile(self.origin_tilt, [self.n, 1])
        self.pixel_spcing = pixel_spacing_mm
        if hasattr(geometry, "name"):
            self.name = geometry.name
        else:
            self.name = "unknown"

        # get slice rotations
        if geometry.tilt_ang_range_rad is None:
            std_p_1 = torch.tensor(
                [
                    self.vh.shape[0] // 2,
                    self.vh.shape[1] + (0.3 * self.vh.shape[1] // 2),
                    self.vh.shape[2],
                ]
            ) * torch.tensor(pixel_spacing_mm)
            std_p_2 = torch.tensor(
                [
                    self.vh.shape[0] // 2,
                    -(0.3 * self.vh.shape[1] // 2),
                    self.vh.shape[2],
                ]
            ) * torch.tensor(pixel_spacing_mm)
            v_1 = self.origin_tilt - std_p_1
            v_2 = self.origin_tilt - std_p_2
            norm_v_1 = v_1 / torch.norm(v_1)
            norm_v_2 = v_2 / torch.norm(v_2)
            angle = torch.arccos_(torch.sum(norm_v_1 * norm_v_2)).item()
        else:
            angle = geometry.tilt_ang_range_rad

        # get affine rotation matrices
        self.thetas = torch.tensor(
            [
                [t_x, t_y, t_z]
                for t_x, t_y, t_z in zip(
                    np.linspace(-angle / 2, angle / 2, self.n),
                    np.linspace(0, 0, self.n),
                    np.linspace(0, 0, self.n),
                )
            ]
        ).to(dtype=torch.float32, device=device)
        pre_rot_mat = (rotmat_from_euler(torch.tensor(pre_rot)[None, ...])[0]).to(device=device)
        affine_mats = rotmat_from_euler(self.thetas)
        scale_mat = torch.tensor(np.eye(4) / np.concatenate([pixel_spacing_mm, [1]]), dtype=torch.float32).to(
            device=device
        )
        unscale_mat = torch.tensor(np.eye(4) * np.concatenate([pixel_spacing_mm, [1]]), dtype=torch.float32).to(
            device=device
        )
        self.affine_mats = scale_mat @ pre_rot_mat @ affine_mats @ unscale_mat

        # Add translation to origines
        self.affine_mats[:, :3, 3].copy_(self.Ts)

        # get coords
        self.slice_coords_cart = self.sh.get_cart_coord_FoV(self.affine_mats)

        # get grey value slices
        self.slices = self.vh.get_pixel_value_cart(self.slice_coords_cart.clone())

        # get mask slices
        slices_mask = self.vh.get_mask_value_cart(self.slice_coords_cart.clone())
        slices_mask[slices_mask > 0.5] = 1
        slices_mask[slices_mask <= 0.5] = 0
        self.slices_mask = slices_mask.to(dtype=bool)

    def __len__(self):
        return self.n

    def __getitem__(self, index) -> tuple:
        return index, self.slices[index], self.slices_mask[index]

    def get_localization(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns translations and rotations about the axis which parametrize the set of slices

        Returns:
            tuple[torch.Tensor, torch.Tensor]: (thetas, translations)
        """
        return self.thetas, self.Ts

    def get_aff_trans_mat(self, idx) -> torch.Tensor:
        """Returns affine transformation matrix of index

        Args:
            idx: Index or list of indices

        Returns:
            torch.Tensor: Affine transformation matrices of index.
        """
        return self.affine_mats[idx]

    def _add_pre_transform(self, aff_trans_mat: torch.Tensor) -> None:
        """Adds a pre transform as class object

        Args:
            aff_trans_mat (torch.Tensor): Pre transform
        """
        self.aff_pre_trans = aff_trans_mat
        return


class Dataset_2D_lin_array(data.Dataset):
    def __init__(
        self,
        stack: np.ndarray,
        pixel_spacing_mm: list[float] = [1.0, 1.0, 1.0],
    ) -> None:
        """Creates pytorch dataset from a 2D slice stack, by sampling evenly spaced slices from the volume.

        Args:
            stack (np.ndarray): Volume to create dataset from.
            model_params (dict): Python dictionary contoaining model parameters
            geometry (dict): Python dictionary containing gemoeric information
            pixel_spacing_mm (list[float], optional): Pixel spacing in mm. Defaults to [1.,1.,1.].
            pre_rot (list[float], optional): Preliminary rotation. Defaults to [0,0,0].
        """

        super().__init__()

        # define geometry
        framsesize = list(stack.shape[:-1])
        self.sh = SliceHandlerLinearProbe(framsesize, [0, 0])

        mask = torch.full(stack.shape, True, dtype=torch.bool)

        # create volume handler
        self.stackh = StackHandler(stack=torch.tensor(stack), pixel_spacing=pixel_spacing_mm, volume_mask=mask)

        self.n = stack.shape[-1]

        # get source shifts
        self.origin_slice = torch.tensor((0, 0))  # in pixels upper left corner of slice
        self.Ts = torch.tile(self.origin_slice, (self.n, 1))
        self.Ts = torch.cat([self.Ts, torch.arange(stack.shape[-1])[..., None]], dim=1)
        self.pixel_spcing = pixel_spacing_mm

        # get slice rotations

        # get affine rotation matrices
        self.thetas = torch.zeros_like(self.Ts, dtype=torch.float32)
        affine_mats = rotmat_from_euler(self.thetas)
        scale_mat = torch.tensor(np.eye(4) / np.concatenate([pixel_spacing_mm, [1]]), dtype=torch.float32)
        unscale_mat = torch.tensor(np.eye(4) * np.concatenate([pixel_spacing_mm, [1]]), dtype=torch.float32)
        self.affine_mats = scale_mat @ affine_mats @ unscale_mat

        # Add translation to origines
        self.affine_mats[:, :3, 3].copy_(self.Ts)

        # get coords
        self.slice_coords_cart = self.sh.get_cart_coord_FoV(self.affine_mats)

        # get grey value slices
        self.slices = self.stackh.get_pixel_value_cart(self.slice_coords_cart.clone())

        # get mask slices
        self.slices_mask = self.stackh.get_mask_value_cart(self.slice_coords_cart.clone())

    def __len__(self):
        return self.n

    def __getitem__(self, index) -> tuple:
        return index, self.slices[index], self.slices_mask[index]

    def get_localization(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns translations and rotations about the axis which parametrize the set of slices

        Returns:
            tuple[torch.Tensor, torch.Tensor]: (thetas, translations)
        """
        return self.thetas, self.Ts

    def get_aff_trans_mat(self, idx) -> torch.Tensor:
        """Returns affine transformation matrix of index

        Args:
            idx: Index or list of indices

        Returns:
            torch.Tensor: Affine transformation matrices of index.
        """
        return self.affine_mats[idx]

    def _add_pre_transform(self, aff_trans_mat: torch.Tensor) -> None:
        """Adds a pre transform as class object

        Args:
            aff_trans_mat (torch.Tensor): Pre transform
        """
        self.aff_pre_trans = aff_trans_mat
        return
