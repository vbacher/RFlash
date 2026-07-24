from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import torch
from typing import Sequence

# FIXME: Remove this global counter in final version which is publicly released. This is only for debugging and visualization of intermediate results.
ctr = 0


# FIXME: remove this global variable in final version which is publicly released. This is only for debugging and visualization of intermediate results.
def visualize_2d_image(
    img: np.ndarray,
    spacing_mm: np.ndarray = np.asarray([1, 1]),
    title="Image",
    f_name=None,
    file=None,
) -> None:
    # FIXME: Remove in final version which is publicly released. This function is only for debugging and visualization of intermediate results.
    """THIS FUNCTION IS ONLY FOR DEBUGGING"""

    if img.ndim == 2:
        height, width = img.shape
    elif img.ndim == 3 and img.shape[2] == 3:
        height, width, _ = img.shape
    else:
        raise ValueError(f"Expected image shape (H, W) or (H, W, 3), got {img.shape}")

    extent = [0, width * spacing_mm[1], height * spacing_mm[0], 0]

    _, ax = plt.subplots()
    ax.set_title(title)
    if img.ndim == 2:
        ax.imshow(img, cmap="grey", extent=extent, origin="upper")
    else:
        ax.imshow(img, extent=extent, origin="upper")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    if file is None:
        file = "./debugging"

    if f_name is None:
        global ctr
        plt.savefig(f"{file}/out_{ctr:03d}.png")
        ctr += 1
    else:
        plt.savefig(f"{file}/{f_name}")
    plt.close()


def visualize_vol(volume: np.ndarray | torch.Tensor, spacing_mm: Sequence = [1, 1, 1], title="volume") -> None:
    """Visualize ultrasound volume. For easier debugging, the visualization is not normalized.

    Args:
        volume (np.ndarray | torch.Tensor): Input volume. Must be 3D
        spacing_mm (Sequence, optional): Pixcel spacing. Array must be of length 3. Defaults to [1,1,1].
        title (str, optional): Title of image. Defaults to "volume".

    Example:
        >>> vol = np.random.randint(0,255,(60,60,60))
        >>> visualize_volume(vol)
    """

    if torch.is_tensor(volume):
        volume = volume.cpu().detach().numpy()

    assert len(volume.shape) == 3, f"this function prints only 2D images but got tensor of shape {volume.shape}"
    assert len(spacing_mm) == len(
        volume.shape
    ), f"Mismatching dimentions of image and scaling. Dimensions image: {volume.shape}; Dimenstions scaling: {spacing_mm}"

    midplanes = [i // 2 for i in volume.shape]
    fig, ax = plt.subplots(1, 3, figsize=(10, 3), layout="constrained")
    fig.suptitle(title)
    ax[0].imshow(volume[midplanes[0]], cmap="grey", aspect=spacing_mm[1] / spacing_mm[2])
    ax[0].set_title("coronal (y-z). Source left")
    ax[1].imshow(volume[:, midplanes[1], :], cmap="grey", aspect=spacing_mm[0] / spacing_mm[2])
    ax[1].set_title("axial (x-z). Source left")
    ax[2].imshow(volume[:, :, midplanes[2]], cmap="grey", aspect=spacing_mm[0] / spacing_mm[1])
    ax[2].set_title("sagittal (x-y). Source top")
    if not REMOTE:
        plt.show()
    else:
        global ctr
        plt.savefig(f"./debugging/out_{ctr:03d}.png")
        ctr += 1
    plt.close()
