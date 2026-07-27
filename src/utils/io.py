from __future__ import annotations

from glob import glob
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from medpy.io.header import Header
from medpy.io.load import load
from medpy.io.save import save

IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}


def get_medpy_header(spacing_mm: list[float] | np.ndarray | None = None) -> Header:
    """Create a minimal MedPy header with voxel spacing for saved volumes."""

    if spacing_mm is None:
        spacing_mm = [1.0, 1.0, 1.0]
    header = Header(spacing=tuple(spacing_mm))
    return header


def load_volume(file_path: str) -> tuple[np.ndarray, Header]:
    """Load a 3D medical image volume from an ``.mha`` file.

    Args:
        file_path: Path to the input ``.mha`` file.

    Returns:
        The image volume and its MedPy header.
    """

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {path}")
    suffix = path.suffix.lower()
    if suffix == ".gz":
        suffix = path.suffixes[-2].lower() + suffix
    if suffix != ".mha" and suffix != ".nii" and suffix != ".nii.gz":
        raise ValueError(f"Unsupported volume format '{path.suffix}'. Only .mha , .nii, and .nii.gz are supported.")

    volume, header = load(str(path))
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D .mha volume, got shape {volume.shape}.")
    return np.asarray(volume), header


def load_image_directory(directory_path: str | Path) -> np.ndarray:
    """Load a directory of 2D grayscale ultrasound images as a stack.

    Args:
        directory_path: Directory containing image files.

    Returns:
        A stack with shape ``(num_images, rows, columns)``.
    """

    path = Path(directory_path)
    if not path.is_dir():
        raise NotADirectoryError(f"Expected an image directory, got: {path}")

    image_paths = glob(str(path / "*.jpg"))
    if not image_paths:
        raise FileNotFoundError(f"No supported image files found in {path}.")

    images = [_read_grayscale_image(image_path) for image_path in image_paths]
    first_shape = images[0].shape
    if any(image.shape != first_shape for image in images):
        raise ValueError("All images in the directory must have the same shape.")
    return np.stack(images, axis=0)


def _ensure_image_stack(array: np.ndarray) -> np.ndarray:
    """Convert common ``.npy`` layouts to ``(num_images, rows, columns)``."""

    image_array = np.asarray(array)
    if image_array.ndim == 2:
        return image_array[None, ...]
    if image_array.ndim == 3:
        return image_array
    if image_array.ndim == 4 and image_array.shape[-1] in (1, 3, 4):
        if image_array.shape[-1] == 1:
            return image_array[..., 0]
        return np.dot(image_array[..., :3], np.asarray([0.299, 0.587, 0.114]))
    raise ValueError(f"Expected 2D images or an image stack, got shape {image_array.shape}.")


def _read_grayscale_image(image_path: Path | str) -> np.ndarray:
    """Read an image with matplotlib and convert RGB/RGBA files to grayscale."""

    image = plt.imread(image_path)
    image_array = np.asarray(image)
    if image_array.ndim == 2:
        return image_array.astype(np.float32)
    if image_array.ndim == 3:
        channels = image_array[..., :3].astype(np.float32)
        return np.dot(channels, np.asarray([0.299, 0.587, 0.114], dtype=np.float32))
    raise ValueError(f"Unsupported image shape {image_array.shape} in {image_path}.")


def normalize_for_display(image: np.ndarray) -> np.ndarray:
    """Normalize an image or volume to the range ``[0, 1]`` for plotting."""

    image_array = np.asarray(image, dtype=float)
    finite_mask = np.isfinite(image_array)
    if not finite_mask.any():
        return np.zeros_like(image_array, dtype=float)

    finite_values = image_array[finite_mask]
    image_min = float(finite_values.min())
    image_max = float(finite_values.max())
    if np.isclose(image_min, image_max):
        return np.zeros_like(image_array, dtype=float)

    normalized = np.nan_to_num(image_array, nan=image_min, posinf=image_max, neginf=image_min)
    return np.clip((normalized - image_min) / (image_max - image_min), 0.0, 1.0)


def load_synthetic_liver(input_path: str | Path) -> np.ndarray:
    """Load synthetic liver ultrasound data from a file or directory of ``.npy`` arrays.

    Args:
        input_path: Either one ``.npy`` file or a directory containing ``.npy``
            files such as ``images-l2.npy`` and ``images-r2.npy``.

    Returns:
        A stack with shape ``(num_images, rows, columns)``.
    """

    path = Path(input_path)
    if path.is_file():
        arrays = [np.load(path)]
    elif path.is_dir():
        npy_paths = sorted(path.glob("*.npy"))
        if not npy_paths:
            raise FileNotFoundError(f"No .npy files found in {path}.")
        arrays = [np.load(npy_path) for npy_path in npy_paths]
    else:
        raise FileNotFoundError(f"Input path does not exist: {path}")

    stacks = [_ensure_image_stack(array) for array in arrays]
    first_shape = stacks[0].shape[1:]
    if any(stack.shape[1:] != first_shape for stack in stacks):
        raise ValueError("Synthetic liver arrays must share the same image shape.")
    return np.transpose(np.concatenate(stacks, axis=0), (1, 2, 0))


def to_8bit_graysacle(image: Any, volume_mask: Any | None = None) -> np.ndarray:

    if torch.is_tensor(image):
        image = image.detach().cpu().numpy()

    if volume_mask is not None:
        image = image - image[volume_mask].mean()
        image = image / (image[volume_mask].std() / (128 / 3))
        image = image + 128
        image[image < 0] = 0
        image[image > 255] = 255
        image = image.astype(np.uint8)
    else:
        image = image - image.min()
        image = ((image / image.max()) * 255).astype(np.uint8)
    return image


def save_img(img, spacing_mm, o_path, f_name, title=None) -> None:
    _, ax = plt.subplots()
    if title is not None:
        ax.set_title(title)
    ax.imshow(img, aspect=spacing_mm[0] / spacing_mm[1], cmap="grey")
    plt.savefig(f"{o_path}/{f_name}.png")
    plt.close()


def save_volume(
    volume: np.ndarray | torch.Tensor,
    file_path: Path | str,
    header: None | Header = None,
) -> None:
    """Helper to save volumes

    Args:
        volume (np.ndarray | torch.Tensor): Volume to store
        opath (Path): opath without filename
        filename (str): Filename
        data_format (str, optional): data format. Defaults to '.mha'.
        header (None | Header, optional): Medpy header. Defaults to None.
        pixel_spacing (list, optional): Pixel spacing. Defaults to [1,1,1].
        override (bool, optional): Silently override existing volumes. Defaults to True.
    """

    if isinstance(file_path, str):
        file_path = Path(file_path)

    data_format = Path(file_path).suffix.lower()
    if data_format == ".gz":
        data_format = Path(file_path).suffixes[-2].lower() + data_format

    assert any(
        data_format == i for i in [".mha", ".nii", ".nii.gz", ".dcm", ".mhd"]
    ), f" The format {data_format} is currently not supported"

    # check if folder exists
    if not file_path.parent.exists():
        file_path.parent.mkdir(parents=True)

    # save file
    save(volume, file_path, hdr=header, use_compression=True)
