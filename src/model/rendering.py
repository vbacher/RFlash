###### imports ######

## library imports
import torch
from matplotlib.image import imsave
from scipy.ndimage import binary_erosion
import numpy as np

## project imports
from src.utils.visualization import visualize_2d_image, visualize_vol
from src.utils.io import save_img, to_8bit_graysacle

from src.utils.geometry import pseudo_inverse_bilinear_interpolation, pseudo_inverse_trilinear_interpolation

###### body ######


class Render_engine(torch.nn.Module):
    def __init__(
        self,
        image_size_polar: list | np.ndarray,
        constant_init_values: list[float] = [0.05, 0.1, 0.05, 1, 1, 1],
        compression: float = 1e-4,
        frequency: float = 20,  # 3.9 / 0.05, with alpha = 0.5
    ) -> None:
        """Constructor of render engine

        Args:
            image_size_polar (list): Image dimensions of slices in polar coordinates
            frequency (float, optional): frenquency used for rendering. Defaults to 2.5.
        """

        super(Render_engine, self).__init__()

        self.image_size_polar = image_size_polar
        self.freqency = frequency

        # define time gain compensation
        alpha_prototype = torch.full(size=(1, image_size_polar[0]), fill_value=constant_init_values[0])[0]
        TGC = torch.cumsum(alpha_prototype / alpha_prototype.shape[0] * self.freqency, dim=0)

        self.TGC = torch.nn.Parameter(TGC, requires_grad=False)
        self.compression = torch.nn.Parameter(torch.tensor(compression), requires_grad=False)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Performs rendering for two parameter maps

        Args:
            input_tensor (torch.Tensor): Input tensor containing parameter maps of shape (batch size, H, W, 2)

        Returns:
            torch.Tensor: Rendered slices of shape (batch size, H, W)
        """

        bs = input_tensor.shape[0]

        # extract parameter maps
        alpha_t = input_tensor[:, :, :, 0]
        # self.visualize_fan_slice(alpha_t, "attenuation")
        phi_t = input_tensor[:, :, :, 1]
        # self.visualize_fan_slice(phi_t, "scatter")

        # signal decay after Labert-Beers-Law
        Integral = torch.cumsum((alpha_t) * self.freqency / alpha_t.shape[1], dim=1)
        Integral[:, 1:, :] = Integral[:, :-1, :].clone()
        Integral[:, 0, :] = 0
        I_t = torch.exp(-Integral)
        # self.visualize_fan_slice(I_t, "intensity")

        # decay scatter
        render = I_t * phi_t
        # self.visualize_fan_slice(render, "measured")

        # apply TGC
        TGC = torch.tile(torch.exp(self.TGC).reshape(-1, 1), (bs, 1, self.image_size_polar[1]))
        render = TGC * render
        # self.visualize_fan_slice(render, "a_tgc")

        render = self.log_compression(render)
        # self.visualize_fan_slice(render, "a_log")

        return render

    def rend_wo_shadow(self, input_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Performs shadow avoiding rendering for two parameter maps.

        Args:
            input_tensor (torch.Tensor): Input tensor containing parameter maps of shape (batch size, H, W, 2)

        Returns:
            tuple[torch.Tensor,torch.Tensor]: Rendered shadow free slices of shape (batch size, H, W) as well as a shadow probability map.
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

        # log compression
        render = self.log_compression(render)

        return render, probab_map

    def log_compression(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log(1 + torch.abs(self.compression) * x) / torch.log(
            1 + torch.abs(self.compression)
        )  # Wein, 2008

    def inverse_log_compression(self, x: torch.Tensor) -> torch.Tensor:
        return (torch.pow(torch.full_like(x, self.compression + 1), x) - 1) / self.compression

    def save_stage_maps(
        self,
        param_maps: torch.Tensor,
        slice_coords: torch.Tensor,
        vol_shape: torch.Tensor,
        pixel_spacing: torch.Tensor,
    ) -> None:

        slice_coords = torch.cat(
            [
                slice_coords[..., 0:1],
                slice_coords[..., 2:3],
                torch.zeros_like(slice_coords[..., 1:2]).to(slice_coords.device),
            ],
            dim=3,
        )

        bs = param_maps.shape[0]

        # extract parameter maps
        alpha_t = param_maps[:, :, :, 0]
        # Ensure the output of pseudo_inverse_bilinear_interpolation is used or validated
        self.__save_simulation_space_slice(alpha_t, slice_coords, vol_shape, pixel_spacing, "./output", "alpha")
        phi_t = param_maps[:, :, :, 1]
        self.__save_simulation_space_slice(phi_t, slice_coords, vol_shape, pixel_spacing, "./output", "phi")

        # signal decay after Labert-Beers-Law
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
