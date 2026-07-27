###### imports ######

## library imports
from collections.abc import Sequence

import numpy as np
import torch
from torch import Tensor, device, float32, nn, tensor

## project imports
from src.model.initialization import Normal_initialization
from src.utils.geometry import bilinear_interpolation_stack, trilinear_interpolation
from src.utils.transformations import rotmat_from_euler

###### body ######


class SlicePoses(nn.Module):
    def __init__(
        self,
        pose: tuple[Tensor, Tensor] | np.ndarray,
        pixel_spacing_mm: list[float] | np.ndarray = [1.0, 1.0, 1.0],
    ) -> None:
        """Initializes a model learning the location and orientation of the slices.

        Args:
            pose (tuple[torch.Tensor]): tuple containing shifts and euler angles of the slices.
            learn_R (bool, optional): Define if rotations are set trainable. Defaults to True.
            learn_t (bool, optional): Define if translations are trainable. Defaults to True.
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
        """_summary_

        Args:
            pose_id (int | Sequence): Index or list of indices

        Returns:
            torch.Tensor: affine transformation matrix
        """
        a = self.r[pose_id]
        r = rotmat_from_euler(a)
        t = self.t[pose_id]
        affine_mat = self.scale_mat @ r @ self.unscale_mat
        affine_mat[:, :3, 3] = t
        return affine_mat

    def get_device(self) -> device:
        """Helper to get device of model

        Returns:
            device: _description_
        """
        return next(self.parameters()).device


class ExplicitRepresentation(torch.nn.Module):
    def __init__(
        self,
        volume_shape: tuple,
        constant_init_values: list[float] = [0.05, 0.1, 0.05, 1, 1, 1],
        stack: bool = False,
    ) -> None:
        """Constrructor of Explicit representation

        Args:
            volume_shape (tuple): shape of cost volume
            constant_init_values (list[float], optional): constant values to initialize cost volume with. One per parameter map. Defaults to [0.05,0.1,0.05,1,1,1].
            dataset (Dataset_3D_volume | None, optional): Dataset object. Defaults to None.
        """

        super().__init__()

        # set model variables
        self.dim_vol = len(volume_shape)
        self.vol_shape = volume_shape
        self.vol_shape_extended = np.asarray(volume_shape) + 6

        # create model parameters
        self.cost_vol = torch.zeros((*self.vol_shape_extended, 1))
        self.cost_vol = torch.tile(self.cost_vol, (*tuple([1 for i in range(self.dim_vol)]), 2))

        # init weights
        self._init_weights(constant_init_values)

        self.cost_vol = torch.nn.Parameter(self.cost_vol.float(), requires_grad=True)
        self.stack = stack

    def _init_weights(
        self,
        constant_init_values: list[float] = [0.05, 0.1, 0.05, 1, 1, 1],
    ) -> None:
        """Initializes weights.

        Args:
            initialization (str, optional): Initilization type.  Currently supports 'norm' and 'hughes'. Defaults to 'norm'.
            constant_init_values (list[float], optional): constant values to initialize cost volume with. One per parameter map. Defaults to [0.05,0.1,0.05,1,1,1].
        """
        assert not self.cost_vol.requires_grad, "weights can only be initialized before setting them trainable"
        init_values = Normal_initialization(
            shape_cost_vol=self.vol_shape,
            constant_init_vals=constant_init_values,
        )
        self.cost_vol[3:-3, 3:-3, 3:-3] = init_values

    def get_cost_vol(self) -> torch.Tensor:
        """Returns the cost volume without padding

        Returns:
            torch.Tensor: _description_
        """
        return self.cost_vol[3:-3, 3:-3, 3:-3]

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """_summary_

        Args:
            input (torch.Tensor): _description_

        Returns:
            torch.Tensor: _description_
        """
        if self.stack:
            parameter_maps = bilinear_interpolation_stack(input, self.cost_vol[:, :, 3:-3, :], padding=False)
        else:
            parameter_maps = trilinear_interpolation(input, self.cost_vol, padding=False)
        parameter_maps[parameter_maps < 10e-7] = 10e-7
        return parameter_maps

    def get_device(self) -> torch.device:
        """Helper to get device of model

        Returns:
            device: device of parameters
        """
        return next(self.parameters()).device
