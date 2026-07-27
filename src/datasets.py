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
    PyTorch dataset adapters that convert input ultrasound volumes or 2D image
    stacks into fixed slice samples for RFlash training.
License:
    This file is part of the RFlash project and is distributed under the
    repository's LICENSE. See the LICENSE file in the repository root for
    licensing information.
------------------------------------------------------------------------------"""

import numpy as np
import torch
from torch.utils import data

from src._datatypes import VolumeTransducerGeometry
from src.trainings_params import Parameter_Demo3D
from src.utils.geometry import (
    SliceHandler,
    SliceHandlerLinearProbe,
    StackHandler,
    VolumeHandler,
)
from src.utils.transformations import rotmat_from_euler


class Dataset_3D_volume(data.Dataset):
    """Dataset of fan slices sampled from one 3D ultrasound volume.

    The dataset precomputes slice coordinates, intensity samples, and binary
    support masks from the estimated scanner geometry. During training, each
    item contains one slice index, the corresponding observed slice, and its
    mask.
    """

    def __init__(
        self,
        volume: np.ndarray,
        model_params: Parameter_Demo3D,
        geometry: VolumeTransducerGeometry,
        pixel_spacing_mm: list[float] | np.ndarray = [1.0, 1.0, 1.0],
        pre_rot: list[float] = [0, 0, 0],
        device: torch.device | None = None,
    ) -> None:
        """Create a dataset by sampling evenly spaced fan slices from a volume.

        Args:
            volume: Input volume with shape ``(height, width, depth)``.
            model_params: Demo parameters containing polar image size and
                number of fan slices.
            geometry: Estimated 3D transducer geometry.
            pixel_spacing_mm: Voxel spacing in millimetres as
                ``(height, width, depth)``.
            pre_rot: Optional fixed Euler-angle pre-rotation in radians.
            device: Device used for precomputed affine matrices.

        Raises:
            AssertionError: If ``model_params.num_fan_slices`` is not set.
        """

        super().__init__()
        if device is None:
            device = torch.device("cpu")

        # The handler keeps interpolation and mask handling separate from the
        # dataset bookkeeping.
        self.vh = VolumeHandler(volume=torch.tensor(volume.copy()), pixel_spacing=pixel_spacing_mm)

        # Define the fan slice geometry in the volume's x-z plane. The middle
        # dimension is introduced later as the tilt dimension.
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

        # All slices share the estimated tilt-source origin; only their
        # rotation angle changes across the fan stack.
        self.origin_tilt = torch.tensor(geometry.tilt_source_pix)
        self.Ts = torch.tile(self.origin_tilt, [self.n, 1])
        self.pixel_spcing = pixel_spacing_mm
        if hasattr(geometry, "name"):
            self.name = geometry.name
        else:
            self.name = "unknown"

        # When no measured tilt angle is available, infer an approximate angle
        # from two reference points spanning the scanner sweep in the volume.
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

        # Scaling into millimetres before rotation keeps anisotropic voxel
        # spacing from distorting the physical slice geometry.
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

        # Add translation to origins.
        self.affine_mats[:, :3, 3].copy_(self.Ts)

        # Precomputing coordinates and sampled intensities keeps the training
        # loop small and avoids repeating deterministic interpolation work.
        self.slice_coords_cart = self.sh.get_cart_coord_FoV(self.affine_mats)

        self.slices = self.vh.get_pixel_value_cart(self.slice_coords_cart.clone())

        slices_mask = self.vh.get_mask_value_cart(self.slice_coords_cart.clone())
        slices_mask[slices_mask > 0.5] = 1
        slices_mask[slices_mask <= 0.5] = 0
        self.slices_mask = slices_mask.to(dtype=bool)

    def __len__(self):
        """Return the number of sampled fan slices."""

        return self.n

    def __getitem__(self, index) -> tuple:
        """Return one training sample.

        Args:
            index: Integer slice index supplied by the data loader.

        Returns:
            Tuple ``(index, slice, mask)`` where ``slice`` and ``mask`` have
            shape ``(radial_samples, angular_samples)``.
        """

        return index, self.slices[index], self.slices_mask[index]

    def get_localization(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return Euler rotations and translations that parametrize the slices.

        Returns:
            Tuple ``(thetas, translations)`` with shape ``(num_slices, 3)`` for
            both tensors.
        """
        return self.thetas, self.Ts

    def get_aff_trans_mat(self, idx) -> torch.Tensor:
        """Return affine transformation matrices for selected slice indices.

        Args:
            idx: Integer index, slice, or tensor/list of indices.

        Returns:
            Affine transformation matrix or matrices with trailing shape
            ``(4, 4)``.
        """
        return self.affine_mats[idx]

    def _add_pre_transform(self, aff_trans_mat: torch.Tensor) -> None:
        """Store an additional pre-transform on the dataset.

        Args:
            aff_trans_mat: Affine pre-transform matrix.
        """
        self.aff_pre_trans = aff_trans_mat


class Dataset_2D_lin_array(data.Dataset):
    """Dataset for a stack of aligned 2D linear-probe slices.

    The stack is represented as a pseudo-volume with slice index along the last
    axis. Each training item returns one image plane and a full foreground mask.
    """

    def __init__(
        self,
        stack: np.ndarray,
        pixel_spacing_mm: list[float] = [1.0, 1.0, 1.0],
    ) -> None:
        """Create a dataset from a 2D image stack.

        Args:
            stack: Input stack with shape ``(height, width, num_slices)``.
            pixel_spacing_mm: Pixel spacing in millimetres as
                ``(height, width, slice)``.
        """

        super().__init__()

        # Linear-probe slices are already Cartesian, so the slice handler simply
        # builds a regular grid and inserts the slice index as the third axis.
        framsesize = list(stack.shape[:-1])
        self.sh = SliceHandlerLinearProbe(framsesize, [0, 0])

        mask = torch.full(stack.shape, True, dtype=torch.bool)

        self.stackh = StackHandler(stack=torch.tensor(stack), pixel_spacing=pixel_spacing_mm, volume_mask=mask)

        self.n = stack.shape[-1]

        # The first two translation coordinates remain at the slice origin; the
        # third coordinate enumerates the input image stack.
        self.origin_slice = torch.tensor((0, 0))  # in pixels upper left corner of slice
        self.Ts = torch.tile(self.origin_slice, (self.n, 1))
        self.Ts = torch.cat([self.Ts, torch.arange(stack.shape[-1])[..., None]], dim=1)
        self.pixel_spcing = pixel_spacing_mm

        # Linear-stack demo poses are fixed identity rotations with per-slice
        # translations along the stack axis.
        self.thetas = torch.zeros_like(self.Ts, dtype=torch.float32)
        affine_mats = rotmat_from_euler(self.thetas)
        scale_mat = torch.tensor(np.eye(4) / np.concatenate([pixel_spacing_mm, [1]]), dtype=torch.float32)
        unscale_mat = torch.tensor(np.eye(4) * np.concatenate([pixel_spacing_mm, [1]]), dtype=torch.float32)
        self.affine_mats = scale_mat @ affine_mats @ unscale_mat

        # Add translation to origins.
        self.affine_mats[:, :3, 3].copy_(self.Ts)

        self.slice_coords_cart = self.sh.get_cart_coord_FoV(self.affine_mats)

        self.slices = self.stackh.get_pixel_value_cart(self.slice_coords_cart.clone())

        self.slices_mask = self.stackh.get_mask_value_cart(self.slice_coords_cart.clone())

    def __len__(self):
        """Return the number of slices in the stack."""

        return self.n

    def __getitem__(self, index) -> tuple:
        """Return one training sample.

        Args:
            index: Integer slice index supplied by the data loader.

        Returns:
            Tuple ``(index, slice, mask)`` where ``slice`` and ``mask`` have
            shape ``(height, width)``.
        """

        return index, self.slices[index], self.slices_mask[index]

    def get_localization(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return Euler rotations and translations that parametrize the stack.

        Returns:
            Tuple ``(thetas, translations)`` with shape ``(num_slices, 3)`` for
            both tensors.
        """
        return self.thetas, self.Ts

    def get_aff_trans_mat(self, idx) -> torch.Tensor:
        """Return affine transformation matrices for selected slice indices.

        Args:
            idx: Integer index, slice, or tensor/list of indices.

        Returns:
            Affine transformation matrix or matrices with trailing shape
            ``(4, 4)``.
        """
        return self.affine_mats[idx]

    def _add_pre_transform(self, aff_trans_mat: torch.Tensor) -> None:
        """Store an additional pre-transform on the dataset.

        Args:
            aff_trans_mat: Affine pre-transform matrix.
        """
        self.aff_pre_trans = aff_trans_mat
