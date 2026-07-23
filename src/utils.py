"""Data loading helpers for the public RFlash demo.

The release repository supports three lightweight input modes: the packaged
fetal brain ``.mha`` volume, a directory of abdominal ultrasound images, and
the synthetic liver ``.npy`` arrays used by the Ultra-NeRF example data.
"""

from __future__ import annotations

from typing import Any

from pathlib import Path

from glob import glob

import matplotlib.pyplot as plt
import numpy as np
from medpy.io.header import Header
from medpy.io.load import load

# FIXME: Remove this global counter in final version which is publicly released. This is only for debugging and visualization of intermediate results.
ctr = 0

IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}


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
    return np.concatenate(stacks, axis=0)


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


def _read_grayscale_image(image_path: Path) -> np.ndarray:
    """Read an image with matplotlib and convert RGB/RGBA files to grayscale."""

    image = plt.imread(image_path)
    image_array = np.asarray(image)
    if image_array.ndim == 2:
        return image_array.astype(np.float32)
    if image_array.ndim == 3:
        channels = image_array[..., :3].astype(np.float32)
        return np.dot(channels, np.asarray([0.299, 0.587, 0.114], dtype=np.float32))
    raise ValueError(f"Unsupported image shape {image_array.shape} in {image_path}.")


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


def to_jsonable(value: Any) -> Any:
    """Convert NumPy values to plain Python objects for readable printing."""

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    return value


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
