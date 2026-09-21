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
    Lightweight matplotlib visualization helpers for debugging image geometry
    and saving public-demo training statistics.
License:
    This file is part of the RFlash project and is distributed under the
    repository's LICENSE. See the LICENSE file in the repository root for
    licensing information.
------------------------------------------------------------------------------"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def visualize_2d_image(
    img: np.ndarray,
    spacing_mm: np.ndarray = np.asarray([1, 1]),
    title="Image",
    f_name=None,
    file=".",
) -> None:
    """Save a 2D image visualization without opening a matplotlib window."""

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
    output_dir = Path(file)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_name = f_name or "image.png"
    plt.savefig(output_dir / output_name)
    plt.close()


def visualize_vol(
    volume: np.ndarray | torch.Tensor,
    spacing_mm: list[float] | np.ndarray = (1.0, 1.0, 1.0),
    title: str = "volume",
    output_path: str | Path = "volume_midplanes.png",
) -> None:
    """Visualize the three middle planes of an ultrasound volume.

    Args:
        volume: Input volume with shape ``(height, width, depth)``.
        spacing_mm: Voxel spacing in millimetres. Must have length 3.
        title: Figure title.
        output_path: PNG path for the visualization.

    Returns:
        None.

    Example:
        >>> vol = np.random.randint(0,255,(60,60,60))
        >>> visualize_volume(vol)
    """

    if torch.is_tensor(volume):
        volume = volume.cpu().detach().numpy()

    assert len(volume.shape) == 3, f"this function visualizes only 3D volumes but got tensor of shape {volume.shape}"
    assert len(spacing_mm) == len(
        volume.shape
    ), f"Mismatching dimensions of image and scaling. Dimensions image: {volume.shape}; dimensions scaling: {spacing_mm}"

    midplanes = [i // 2 for i in volume.shape]
    fig, ax = plt.subplots(1, 3, figsize=(10, 3), layout="constrained")
    fig.suptitle(title)
    ax[0].imshow(volume[midplanes[0]], cmap="grey", aspect=spacing_mm[1] / spacing_mm[2])
    ax[0].set_title("coronal (y-z). Source left")
    ax[1].imshow(volume[:, midplanes[1], :], cmap="grey", aspect=spacing_mm[0] / spacing_mm[2])
    ax[1].set_title("axial (x-z). Source left")
    ax[2].imshow(volume[:, :, midplanes[2]], cmap="grey", aspect=spacing_mm[0] / spacing_mm[1])
    ax[2].set_title("sagittal (x-y). Source top")
    saved_path = Path(output_path)
    saved_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(saved_path)
    plt.close()


def visualize_stats(
    losses: list[float], l2s: list[float], ssims: list[float], output_dir: str | Path, title="Training Statistics"
) -> None:
    """Save training loss curves as a PNG file.

    Args:
        losses: Epoch-wise total losses.
        l2s: Epoch-wise L2 losses.
        ssims: Epoch-wise SSIM values.
        output_dir: Directory where ``training_stats.png`` is written.
        title: Figure title.

    Returns:
        None.
    """
    fig, ax = plt.subplots(3, 1, figsize=(10, 10), layout="constrained")
    fig.suptitle(title)
    ax[0].plot(losses)
    ax[0].set_title("Loss")
    ax[0].set_yscale("log")
    ax[1].plot(l2s)
    ax[1].set_title("L2")
    ax[1].set_yscale("log")
    ax[2].plot(ssims)
    ax[2].set_title("SSIM")
    ax[2].set_yscale("log")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path / "training_stats.png")
    plt.close()
