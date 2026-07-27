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
    PyTorch modules for fixed slice poses and the trainable explicit RFlash
    attenuation/scatter representation.
License:
    This file is part of the RFlash project and is distributed under the
    repository's LICENSE. See the LICENSE file in the repository root for
    licensing information.
------------------------------------------------------------------------------"""

from collections.abc import Sequence

import numpy as np
import torch
from torch import Tensor, device, float32, nn, tensor

from src.model.initialization import Normal_initialization
from src.utils.geometry import bilinear_interpolation_stack, trilinear_interpolation
from src.utils.transformations import rotmat_from_euler


class SlicePoses(nn.Module):
    """Fixed slice-pose model returning affine matrices for slice indices.

    The public demo does not optimize slice poses. Keeping poses in an
    ``nn.Module`` still makes device placement and batching consistent with the
    trainable representation.
    """

    def __init__(
        self,
        pose: tuple[Tensor, Tensor] | np.ndarray,
        pixel_spacing_mm: list[float] | np.ndarray = [1.0, 1.0, 1.0],
    ) -> None:
        """Initialize slice rotations and translations.

        Args:
            pose: Tuple ``(thetas, translations)``. Both tensors should have
                shape ``(num_slices, 3)`` and contain Euler angles in radians
                and translations in pixels.
            pixel_spacing_mm: Pixel spacing used to scale rotations into
                physical space before returning affine matrices.
        """
        super().__init__()
        thetas = pose[0]
        Ts = pose[1]
        self.scale_mat = nn.Parameter(
            tensor(np.eye(4) / np.concatenate([pixel_spacing_mm, [1]]), dtype=float32), requires_grad=False
        )
        self.unscale_mat = nn.Parameter(
            tensor(np.eye(4) * np.concatenate([pixel_spacing_mm, [1]]), dtype=float32), requires_grad=False
        )
        self.num_cams = len(thetas)
        self.r = nn.Parameter(thetas.float(), requires_grad=False)  # (N, 3)
        self.t = nn.Parameter(Ts.float(), requires_grad=False)  # (N, 3)
        self.r_orig = nn.Parameter(thetas.float(), requires_grad=False)  # (N, 3)
        self.t_orig = nn.Parameter(Ts.float(), requires_grad=False)  # (N, 3)
        self.r_truth = nn.Parameter(thetas.float(), requires_grad=False)  # (N, 3)
        self.t_truth = nn.Parameter(Ts.float(), requires_grad=False)  # (N, 3)

    def forward(self, pose_id: int | Sequence) -> Tensor:
        """Return affine matrices for selected pose indices.

        Args:
            pose_id: Integer index, slice, list, or tensor of indices.

        Returns:
            Affine transformation matrices with shape ``(batch, 4, 4)``.
        """
        a = self.r[pose_id]
        r = rotmat_from_euler(a)
        t = self.t[pose_id]
        affine_mat = self.scale_mat @ r @ self.unscale_mat
        affine_mat[:, :3, 3] = t
        return affine_mat

    def get_device(self) -> device:
        """Return the device that stores the pose parameters.

        Returns:
            PyTorch device for this module.
        """
        return next(self.parameters()).device


class ExplicitRepresentation(torch.nn.Module):
    """Trainable explicit volume of RFlash physical parameter maps.

    The representation stores a padded dense grid whose last axis contains two
    learned maps: attenuation and scatter. Padding keeps interpolation near the
    original volume boundary well-defined.
    """

    def __init__(
        self,
        volume_shape: tuple,
        constant_init_values: list[float] = [0.05, 0.1, 0.05, 1, 1, 1],
        stack: bool = False,
    ) -> None:
        """Construct an explicit padded parameter volume.

        Args:
            volume_shape: Spatial shape of the input volume or stack.
            constant_init_values: Constants used to initialize the parameter
                maps. The first two values initialize attenuation and scatter.
            stack: If true, sample the representation with bilinear stack
                interpolation rather than full trilinear interpolation.
        """

        super().__init__()

        self.dim_vol = len(volume_shape)
        self.vol_shape = volume_shape
        self.vol_shape_extended = np.asarray(volume_shape) + 6

        # Three voxels of padding on each side match the interpolation helpers'
        # coordinate convention.
        self.cost_vol = torch.zeros((*self.vol_shape_extended, 1))
        self.cost_vol = torch.tile(self.cost_vol, (*tuple([1 for i in range(self.dim_vol)]), 2))

        self._init_weights(constant_init_values)

        self.cost_vol = torch.nn.Parameter(self.cost_vol.float(), requires_grad=True)
        self.stack = stack

    def _init_weights(
        self,
        constant_init_values: list[float] = [0.05, 0.1, 0.05, 1, 1, 1],
    ) -> None:
        """Initialize the trainable parameter maps before enabling gradients.

        Args:
            constant_init_values: Constants used by
                :func:`src.model.initialization.Normal_initialization`.

        Raises:
            AssertionError: If called after ``cost_vol`` has been made trainable.
        """
        assert not self.cost_vol.requires_grad, "weights can only be initialized before setting them trainable"
        init_values = Normal_initialization(
            shape_cost_vol=self.vol_shape,
            constant_init_vals=constant_init_values,
        )
        self.cost_vol[3:-3, 3:-3, 3:-3] = init_values

    def get_cost_vol(self) -> torch.Tensor:
        """Return the learned parameter maps without interpolation padding.

        Returns:
            Tensor with shape ``(*volume_shape, 2)``.
        """
        return self.cost_vol[3:-3, 3:-3, 3:-3]

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Sample attenuation/scatter parameter maps at slice coordinates.

        Args:
            input: Cartesian coordinates with shape ``(batch, height, width, 3)``.

        Returns:
            Parameter maps with shape ``(batch, height, width, 2)``.
        """
        if self.stack:
            parameter_maps = bilinear_interpolation_stack(input, self.cost_vol[:, :, 3:-3, :], padding=False)
        else:
            parameter_maps = trilinear_interpolation(input, self.cost_vol, padding=False)
        parameter_maps[parameter_maps < 10e-7] = 10e-7
        return parameter_maps

    def get_device(self) -> torch.device:
        """Return the device that stores the trainable parameter volume.

        Returns:
            PyTorch device for this module.
        """
        return next(self.parameters()).device
