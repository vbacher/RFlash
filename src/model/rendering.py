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
    Differentiable ultrasound renderer used to train and re-render the RFlash
    attenuation/scatter representation.
License:
    This file is part of the RFlash project and is distributed under the
    repository's LICENSE. See the LICENSE file in the repository root for
    licensing information.
------------------------------------------------------------------------------"""

import numpy as np
import torch
from matplotlib.image import imsave
from scipy.ndimage import binary_erosion

from src.utils.geometry import (
    pseudo_inverse_bilinear_interpolation,
    pseudo_inverse_trilinear_interpolation,
)
from src.utils.io import save_img, to_8bit_graysacle


class Render_engine(torch.nn.Module):
    """Differentiable renderer for ultrasound slices.

    The renderer expects two parameter maps per sampled point: attenuation
    ``alpha`` and scatter ``phi``. It models exponential signal decay along the
    scanline, applies a fixed time-gain compensation curve, and finally applies
    logarithmic compression.
    """

    def __init__(
        self,
        image_size_polar: list | np.ndarray,
        constant_init_values: list[float] = [0.05, 0.1, 0.05, 1, 1, 1],
        compression: float = 1e-4,
        frequency: float = 20,  # 3.9 / 0.05, with alpha = 0.5
    ) -> None:
        """Initialize renderer constants.

        Args:
            image_size_polar: Slice dimensions in simulation coordinates as
                ``(radial_samples, angular_samples)``.
            constant_init_values: Initialization constants; the first value is
                used to define the fixed time-gain compensation profile.
            compression: Log-compression parameter.
            frequency: Effective rendering frequency used in the attenuation
                integral.
        """

        super().__init__()

        self.image_size_polar = image_size_polar
        self.freqency = frequency

        # The TGC profile is fixed, not learned. It compensates the renderer's
        # depth-dependent attenuation so the trainable maps can focus on
        # attenuation/scatter decomposition.
        alpha_prototype = torch.full(size=(1, image_size_polar[0]), fill_value=constant_init_values[0])[0]
        TGC = torch.cumsum(alpha_prototype / alpha_prototype.shape[0] * self.freqency, dim=0)

        self.TGC = torch.nn.Parameter(TGC, requires_grad=False)
        self.compression = torch.nn.Parameter(torch.tensor(compression), requires_grad=False)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Render ultrasound slices from attenuation and scatter maps.

        Args:
            input_tensor: Parameter maps with shape ``(batch, height, width, 2)``.

        Returns:
            Rendered slices with shape ``(batch, height, width)``.
        """

        bs = input_tensor.shape[0]

        alpha_t = input_tensor[:, :, :, 0]
        phi_t = input_tensor[:, :, :, 1]

        # Shift the cumulative integral by one sample so that attenuation at a
        # given depth affects subsequent samples rather than itself.
        Integral = torch.cumsum((alpha_t) * self.freqency / alpha_t.shape[1], dim=1)
        Integral[:, 1:, :] = Integral[:, :-1, :].clone()
        Integral[:, 0, :] = 0
        I_t = torch.exp(-Integral)

        render = I_t * phi_t

        TGC = torch.tile(torch.exp(self.TGC).reshape(-1, 1), (bs, 1, self.image_size_polar[1]))
        render = TGC * render

        render = self.log_compression(render)

        return render

    def rend_wo_shadow(self, input_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Render slices without attenuation and estimate the shadow mask.

        Args:
            input_tensor: Parameter maps with shape ``(batch, height, width, 2)``.

        Returns:
            Tuple ``(shadow_free_render, shadow_probability)`` with both tensors
            shaped ``(batch, height, width)``.
        """

        bs = input_tensor.shape[0]

        alpha_t = input_tensor[:, :, :, 0]
        phi_t = input_tensor[:, :, :, 1]

        render = phi_t

        Integral = torch.cumsum((alpha_t) * self.freqency / alpha_t.shape[1], dim=1)
        Integral[:, 1:, :] = Integral[:, :-1, :].clone()
        Integral[:, 0, :] = 0

        I_t = torch.exp(-Integral)
        TGC = torch.tile(torch.exp(self.TGC).reshape(-1, 1), (bs, 1, self.image_size_polar[1]))
        probab_map = 1 - torch.exp(-1 * TGC * I_t)

        render = self.log_compression(render)

        return render, probab_map

    def log_compression(self, x: torch.Tensor) -> torch.Tensor:
        """Apply logarithmic compression to simulated intensities.

        Args:
            x: Non-negative intensity tensor.

        Returns:
            Log-compressed tensor with the same shape as ``x``.
        """

        return torch.log(1 + torch.abs(self.compression) * x) / torch.log(
            1 + torch.abs(self.compression)
        )  # Wein, 2008

    def inverse_log_compression(self, x: torch.Tensor) -> torch.Tensor:
        """Invert :meth:`log_compression` for a compressed intensity tensor.

        Args:
            x: Log-compressed tensor.

        Returns:
            Tensor in the renderer's uncompressed intensity domain.
        """

        return (torch.pow(torch.full_like(x, self.compression + 1), x) - 1) / self.compression

    def save_stage_maps(
        self,
        param_maps: torch.Tensor,
        slice_coords: torch.Tensor,
        vol_shape: torch.Tensor,
        pixel_spacing: torch.Tensor,
    ) -> None:
        """Save intermediate renderer maps for one batch of slices.

        Args:
            param_maps: Parameter maps with shape ``(batch, height, width, 2)``.
            slice_coords: Cartesian sample coordinates for the same batch.
            vol_shape: Target simulation-space volume shape.
            pixel_spacing: Pixel spacing used for image aspect ratio.

        Returns:
            None. Images are written to ``./output``.
        """

        slice_coords = torch.cat(
            [
                slice_coords[..., 0:1],
                slice_coords[..., 2:3],
                torch.zeros_like(slice_coords[..., 1:2]).to(slice_coords.device),
            ],
            dim=3,
        )

        bs = param_maps.shape[0]

        alpha_t = param_maps[:, :, :, 0]
        self.__save_simulation_space_slice(alpha_t, slice_coords, vol_shape, pixel_spacing, "./output", "alpha")
        phi_t = param_maps[:, :, :, 1]
        self.__save_simulation_space_slice(phi_t, slice_coords, vol_shape, pixel_spacing, "./output", "phi")

        # Keep the stage computation identical to ``forward`` so saved maps can
        # be compared directly against rendered training batches.
        Integral = torch.cumsum((alpha_t) * self.freqency / alpha_t.shape[1], dim=1)
        Integral[:, 1:, :] = Integral[:, :-1, :].clone()
        Integral[:, 0, :] = 0
        I_t = torch.exp(-Integral)
        self.__save_simulation_space_slice(I_t, slice_coords, vol_shape, pixel_spacing, "./output", "Intensity")

        # decay scatter
        render = I_t * phi_t
        self.__save_simulation_space_slice(render, slice_coords, vol_shape, pixel_spacing, "./output", "pre_tgc")

        # apply TGC
        TGC = torch.tile(torch.exp(self.TGC).reshape(-1, 1), (bs, 1, self.image_size_polar[1]))
        self.__save_simulation_space_slice(TGC, slice_coords, vol_shape, pixel_spacing, "./output", "tgc")

        # save shadow map
        probab_map = 1 - torch.exp(-1 * TGC * I_t)
        probab_map = probab_map - probab_map.min()
        probab_map = probab_map / probab_map.max()
        probab_map_inverted = probab_map * -1
        probab_map_inverted = probab_map_inverted + 1
        self.__save_simulation_space_slice(
            probab_map,
            slice_coords,
            vol_shape,
            pixel_spacing,
            "./output",
            "shadow_mask",
        )
        self.__save_simulation_space_slice(
            probab_map_inverted,
            slice_coords,
            vol_shape,
            pixel_spacing,
            "./output",
            "shadow_mask_inverted",
        )

        render = TGC * render
        self.__save_simulation_space_slice(render, slice_coords, vol_shape, pixel_spacing, "./output", "pre_log")

        render = self.log_compression(render)
        self.__save_simulation_space_slice(render, slice_coords, vol_shape, pixel_spacing, "./output", "simulated")

        render_sr = self.log_compression(phi_t)
        self.__save_simulation_space_slice(
            render_sr,
            slice_coords,
            vol_shape,
            pixel_spacing,
            "./output",
            "simulated_sr",
        )

    def __save_simulation_space_slice(self, slice, slice_coords, vol_shape, pixel_spacing, o_path, f_name):
        """Project and save one simulation-space slice map as a PNG image."""

        out = pseudo_inverse_bilinear_interpolation(
            slice_coords,
            vol_shape + [1],
            slice,
        )[..., 0]

        save_img(
            out.permute(1, 0).detach().cpu().numpy(),
            pixel_spacing[::-1],
            o_path,
            f_name,
        )

    def visualize_fan_slice(self, fan: torch.Tensor, title="out") -> None:
        """Save a diagnostic visualization of fan slices in volume space.

        Args:
            fan: Fan-space tensor with shape ``(num_slices, height, width)``.
            title: Output filename stem in the ``debugging`` directory.

        Returns:
            None.
        """

        device = fan.device

        # get index of example slice
        idxs = torch.arange(self.pose.num_cams)[60:80]

        with torch.no_grad():
            # retrieve coordinates of example slice
            aff_mat_slice = self.pose(idxs).to(device=device)
            slice_coords = self.dataset.sh.get_cart_coord_FoV(aff_mat_slice).to(device=device)
            shape = self.dataset.vh.shape
            # shape[0] = 1
            vol = pseudo_inverse_trilinear_interpolation(slice_coords, shape, fan[60:80]).cpu()
            volume_mask = vol > 0
            volume_mask = binary_erosion(volume_mask, iterations=2)
            vol = to_8bit_graysacle(vol, volume_mask)

        imsave(f"./debugging/{title}.png", vol[:, 80], cmap="gray")
        del vol, slice_coords, aff_mat_slice
