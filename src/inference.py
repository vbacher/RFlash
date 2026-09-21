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
    Reusable inference routines shared by the command line demo and the Gradio
    web interface.
License:
    This file is part of the RFlash project and is distributed under the
    repository's LICENSE. See the LICENSE file in the repository root for
    licensing information.
------------------------------------------------------------------------------"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src._datatypes import SliceTransducerGeometry, VolumeTransducerGeometry
from src.datasets import Dataset_2D_lin_array, Dataset_3D_volume
from src.model.rendering import Render_engine
from src.model.representation import ExplicitRepresentation, SlicePoses
from src.shadow_reduction import render_volume
from src.train import train_model
from src.trainings_params import (
    Parameter_Demo2D_curvylinear,
    Parameter_Demo2D_linear,
    Parameter_Demo3D,
)
from src.utils.io import (
    get_medpy_header,
    load_image_directory,
    load_image_files,
    load_synthetic_liver,
    load_volume,
    save_volume,
    to_8bit_graysacle,
)
from src.utils.transducer_geometry import (
    estimate_scanner_geometries_stack,
    estimate_scanner_geometry_volume,
)
from src.utils.transformations import (
    resample_to_image_space,
    resample_to_simulation_space,
    standardize_volume,
)
from src.utils.visualization import visualize_stats

DATASET_CHOICES = ("fetal_brain", "abdominal", "synthetic_liver")


@dataclass
class RFlashOutput:
    """Paths written by one RFlash inference run.

    Attributes:
        shadow_reduced_path: Saved shadow-reduced output volume.
        original_path: Saved original input volume in the output image space.
        output_dir: Root output directory used for the run.
        shadow_reduced_data: Processed output array in image space.
        original_data: Original array saved alongside the processed output.
        header: Optional MedPy header used for volume exports.
        spacing_mm: Output voxel spacing.
        is_volume: Whether the result should be treated as a volume preview.
    """

    shadow_reduced_path: Path
    original_path: Path
    output_dir: Path
    shadow_reduced_data: np.ndarray
    original_data: np.ndarray
    header: Any | None
    spacing_mm: np.ndarray
    is_volume: bool


def select_device(prefer_cuda: bool = False) -> torch.device:
    """Select the best available PyTorch device for an inference run.

    Args:
        prefer_cuda: Whether CUDA should be preferred over MPS when both are
            available. The CLI keeps its historical MPS-first behaviour, while
            the web app uses CUDA first for Linux deployment.

    Returns:
        Selected PyTorch device.
    """

    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def run_3D_volume_demo(
    args: argparse.Namespace,
    device: torch.device,
    geometry: VolumeTransducerGeometry | None = None,
) -> Path:
    """Run the full RFlash demo for a 3D ultrasound volume.

    Args:
        args: Parsed command line arguments. ``args.input`` must point to a
            supported 3D volume file, and ``args.output`` selects the output
            directory.
        device: PyTorch device used for model training and rendering.
        geometry: Optional pre-estimated scanner geometry. When omitted, the
            geometry is estimated from the input volume as in the original CLI.

    Returns:
        The path to the shadow-reduced volume.
    """

    return run_3d_volume_inference(
        input_path=args.input,
        output_dir=args.output,
        device=device,
        silent=args.silent,
        geometry=geometry,
    ).shadow_reduced_path


def run_3d_volume_inference(
    input_path: str | Path,
    output_dir: str | Path,
    device: torch.device,
    silent: bool = False,
    geometry: VolumeTransducerGeometry | None = None,
    training_overrides: dict[str, float | int] | None = None,
    progress_callback: Callable[[dict], None] | None = None,
) -> RFlashOutput:
    """Run RFlash on a 3D curvilinear ultrasound volume.

    Args:
        input_path: Path to a supported 3D medical image file.
        output_dir: Directory where outputs should be written.
        device: PyTorch device used for training and rendering.
        silent: Whether to suppress non-essential plots.
        geometry: Optional pre-estimated scanner geometry.
        training_overrides: Optional training-parameter overrides used by the
            web interface.
        progress_callback: Optional training progress callback.

    Returns:
        Paths written by the inference run.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    volume, header = load_volume(input_path)
    spacing_mm = np.asarray([float(value) for value in header.spacing])

    if geometry is None:
        geometry = estimate_scanner_geometry_volume(
            volume,
            spacing_mm,
            overlay_dir=f"{output_dir}/intermediate_images",
            verbose=not silent,
        )

    params = Parameter_Demo3D()
    params.image_size_polar = (np.asarray(volume.shape[::-2]) * 0.9).astype(np.int16)
    params.image_size_cartesian = (np.asarray(volume.shape[::-2]) * 0.9).astype(np.int16)
    params.num_fan_slices = int(volume.shape[1] * 0.9)
    _apply_training_overrides(params, training_overrides)

    volume = standardize_volume(volume, params.init_values[1])

    dataset = Dataset_3D_volume(
        volume=volume, model_params=params, geometry=geometry, pixel_spacing_mm=spacing_mm, device=device
    )

    pose_model = SlicePoses(dataset.get_localization(), spacing_mm)
    representation_model = ExplicitRepresentation(
        volume_shape=volume.shape,
        constant_init_values=params.init_values,
    )
    render_model = Render_engine(
        image_size_polar=dataset.sh.frame_size_pol,
        constant_init_values=params.init_values,
        compression=params.compression,
    )

    loss, l2, ssim = train_model(
        representation_model=representation_model,
        pose_model=pose_model,
        render_model=render_model,
        data=dataset,
        training_params=params,
        device=device,
        progress_callback=progress_callback,
    )
    if not silent:
        visualize_stats(loss, l2, ssim, output_dir=output_dir.joinpath("training_stats"))

    shadow_reduced, _ = render_volume(
        representation_model=representation_model,
        pose_model=pose_model,
        render_model=render_model,
        dataset=dataset,
        batch_size=params.params_generator["batch_size"],
    )

    shadow_reduced_path = output_dir.joinpath("shadow_removed", "shadow_reduced_volume.nii.gz")
    original_path = output_dir.joinpath("shadow_removed", "original_volume.nii.gz")
    save_volume(
        volume=to_8bit_graysacle(shadow_reduced, volume_mask=shadow_reduced > 0.01),
        file_path=shadow_reduced_path,
        header=header,
    )
    save_volume(
        volume=to_8bit_graysacle(volume, volume_mask=volume > 0.01),
        file_path=original_path,
        header=header,
    )

    return RFlashOutput(
        shadow_reduced_path=shadow_reduced_path,
        original_path=original_path,
        output_dir=output_dir,
        shadow_reduced_data=shadow_reduced,
        original_data=volume,
        header=header,
        spacing_mm=spacing_mm,
        is_volume=True,
    )


def run_2D_stack_demo(
    args: argparse.Namespace,
    device: torch.device,
    geometry: list[SliceTransducerGeometry] | None = None,
    indices: list[int] | None = None,
    num_slices: int | None = None,
) -> Path:
    """Run the RFlash demo for a stack of curvilinear 2D ultrasound images.

    Args:
        args: Parsed command line arguments. ``args.input`` must be a directory
            of grayscale-compatible image files.
        device: PyTorch device used for model training and rendering.
        geometry: Optional pre-estimated per-slice scanner geometries.
        indices: Optional original stack indices corresponding to ``geometry``.
        num_slices: Optional number of slices to process in non-interactive
            callers. ``None`` preserves the CLI prompt behaviour.

    Returns:
        The path to the shadow-reduced volume.
    """

    return run_curvilinear_stack_inference(
        input_path=args.input,
        output_dir=args.output,
        device=device,
        silent=args.silent,
        geometry=geometry,
        indices=indices,
        num_slices=num_slices,
    ).shadow_reduced_path


def run_curvilinear_stack_inference(
    input_path: str | Path | list[str | Path],
    output_dir: str | Path,
    device: torch.device,
    silent: bool = False,
    geometry: list[SliceTransducerGeometry] | None = None,
    indices: list[int] | None = None,
    num_slices: int | None = None,
    training_overrides: dict[str, float | int] | None = None,
    progress_callback: Callable[[dict], None] | None = None,
) -> RFlashOutput:
    """Run RFlash on a stack of curvilinear 2D ultrasound images.

    Args:
        input_path: Directory of images, or explicit image file paths.
        output_dir: Directory where outputs should be written.
        device: PyTorch device used for training and rendering.
        silent: Whether to suppress non-essential plots.
        geometry: Optional pre-estimated scanner geometry for selected slices.
        indices: Original stack indices for ``geometry``.
        num_slices: Number of slices to process when geometry is estimated in
            non-interactive contexts.
        training_overrides: Optional training-parameter overrides used by the
            web interface.
        progress_callback: Optional training progress callback.

    Returns:
        Paths written by the inference run.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stack = _load_image_stack(input_path)
    if geometry is None or indices is None:
        geometry, indices = estimate_scanner_geometries_stack(
            stack,
            spacing_mm=(1.0, 1.0),
            overlay_dir=f"{output_dir}/intermediate_images",
            verbose=not silent,
            num_slices=num_slices,
        )
    stack = stack[indices]
    stack = np.transpose(stack, (1, 2, 0))

    params = Parameter_Demo2D_curvylinear()
    params.image_size_polar = (np.asarray(stack.shape[:-1]) * 1.1).astype(np.int16)
    params.image_size_cartesian = np.asarray(stack.shape[:-1]).astype(np.int16)
    _apply_training_overrides(params, training_overrides)

    stack_sim = resample_to_simulation_space(stack, geometry, params)
    vol_mask = resample_to_image_space(np.ones_like(stack, dtype=bool), geometry, params).astype(bool)
    stack_sim = standardize_volume(stack_sim, params.init_values[1])

    output = _run_linear_stack_pipeline(
        stack=stack_sim,
        output_dir=output_dir,
        device=device,
        silent=silent,
        params=params,
        original_stack=stack,
        volume_mask=vol_mask,
        geometry=geometry,
        resample_output=True,
        progress_callback=progress_callback,
    )
    return output


def run_linear_stack_inference(
    input_path: str | Path | list[str | Path],
    output_dir: str | Path,
    device: torch.device,
    silent: bool = False,
    synthetic_liver: bool = False,
    training_overrides: dict[str, float | int] | None = None,
    progress_callback: Callable[[dict], None] | None = None,
) -> RFlashOutput:
    """Run RFlash on linear-probe 2D ultrasound images.

    Args:
        input_path: Image files, an image directory, or synthetic liver ``.npy``
            input when ``synthetic_liver`` is true.
        output_dir: Directory where outputs should be written.
        device: PyTorch device used for training and rendering.
        silent: Whether to suppress non-essential plots.
        synthetic_liver: Whether to use the existing synthetic liver loader.
        training_overrides: Optional training-parameter overrides used by the
            web interface.
        progress_callback: Optional training progress callback.

    Returns:
        Paths written by the inference run.
    """

    if synthetic_liver or _is_npy_input(input_path):
        stack = load_synthetic_liver(input_path)
    else:
        stack = np.transpose(_load_image_stack(input_path), (1, 2, 0))

    params = Parameter_Demo2D_linear()
    _apply_training_overrides(params, training_overrides)
    stack = standardize_volume(stack, params.init_values[1])
    return _run_linear_stack_pipeline(
        stack=stack,
        output_dir=output_dir,
        device=device,
        silent=silent,
        params=params,
        original_stack=stack,
        volume_mask=np.ones_like(stack, dtype=bool),
        progress_callback=progress_callback,
    )


def run_synthetic_liver_demo(args: argparse.Namespace, device: torch.device) -> Path:
    """Run the RFlash demo for synthetic linear-probe liver ultrasound data.

    Args:
        args: Parsed command line arguments. ``args.input`` can be a single
            ``.npy`` file or a directory of compatible ``.npy`` files.
        device: PyTorch device used for model training and rendering.

    Returns:
        The path to the shadow-reduced volume.
    """

    return run_linear_stack_inference(
        input_path=args.input,
        output_dir=args.output,
        device=device,
        silent=args.silent,
        synthetic_liver=True,
    ).shadow_reduced_path


def run_demo(args: argparse.Namespace) -> None:
    """Dispatch the command line demo for the selected dataset.

    Args:
        args: Parsed command line arguments from :func:`demo.parse_args`.

    Returns:
        None.

    Raises:
        ValueError: If ``args.dataset`` is not one of ``DATASET_CHOICES``.
    """

    device = select_device(prefer_cuda=False)
    if not args.silent:
        print(f"executing on {device} device")

    if args.dataset == "fetal_brain":
        o_path = run_3D_volume_demo(args, device=device)
    elif args.dataset == "abdominal":
        o_path = run_2D_stack_demo(args, device=device)
    elif args.dataset == "synthetic_liver":
        o_path = run_synthetic_liver_demo(args, device=device)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    output_mha = getattr(args, "output_mha", None)
    if output_mha is not None:
        output_mha = Path(output_mha)
        if output_mha.suffix.lower() != ".mha":
            raise ValueError("--output-mha must end in .mha.")
        shadow_reduced, header = load_volume(str(o_path))
        save_volume(shadow_reduced, output_mha, header=header)
        print(f"\nDemo finished. Shadow-reduced .mha written to {output_mha}.")
        return

    print(f"\nDemo finished. Outputs written to {o_path}.")


def _load_image_stack(input_path: str | Path | list[str | Path]) -> np.ndarray:
    """Load image data from a directory or explicit uploaded paths."""

    if isinstance(input_path, list):
        return load_image_files(input_path)

    path = Path(input_path)
    if path.is_dir():
        if any(path.glob("*.npy")):
            return np.transpose(load_synthetic_liver(path), (2, 0, 1))
        return load_image_directory(path)
    if _is_volume_input(path):
        volume, _ = load_volume(path)
        return _volume_to_stack(volume)
    return load_image_files([path])


def _is_npy_input(input_path: str | Path | list[str | Path]) -> bool:
    """Return whether an input path should be loaded as NumPy stack data."""

    if isinstance(input_path, list):
        return False
    path = Path(input_path)
    if path.is_file():
        return path.suffix.lower() == ".npy"
    if path.is_dir():
        return any(path.glob("*.npy"))
    return False


def _is_volume_input(input_path: str | Path) -> bool:
    """Return whether a path should be treated as a medical-image volume."""

    path = Path(input_path)
    suffix = path.suffix.lower()
    if suffix == ".gz" and len(path.suffixes) >= 2:
        suffix = path.suffixes[-2].lower() + suffix
    return suffix in {".mha", ".nii", ".nii.gz"}


def _volume_to_stack(volume: np.ndarray) -> np.ndarray:
    """Interpret a saved stack volume as ``(num_slices, rows, columns)``."""

    volume_array = np.asarray(volume)
    if volume_array.ndim != 3:
        raise ValueError(f"Expected a 3D stack volume, got shape {volume_array.shape}.")
    return np.transpose(volume_array, (2, 0, 1))


def _run_linear_stack_pipeline(
    stack: np.ndarray,
    output_dir: str | Path,
    device: torch.device,
    silent: bool,
    params: Parameter_Demo2D_linear | Parameter_Demo2D_curvylinear,
    original_stack: np.ndarray,
    volume_mask: np.ndarray,
    geometry: list[SliceTransducerGeometry] | None = None,
    resample_output: bool = False,
    progress_callback: Callable[[dict], None] | None = None,
) -> RFlashOutput:
    """Train and render the shared linear-stack RFlash pipeline.

    Args:
        stack: Image stack in renderer simulation space with shape
            ``(rows, columns, num_images)``.
        output_dir: Directory where outputs should be written.
        device: PyTorch device used for training and rendering.
        silent: Whether to suppress non-essential plots.
        params: Demo training parameters.
        original_stack: Original image-space stack for saving.
        volume_mask: Foreground mask used for output normalization.
        geometry: Geometry used to map curvilinear results back to image space.
        resample_output: Whether the rendered stack should be resampled back to
            curvilinear image space.
        progress_callback: Optional training progress callback.

    Returns:
        Paths written by the inference run.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = Dataset_2D_lin_array(stack=stack)

    pose_model = SlicePoses(dataset.get_localization())
    representation_model = ExplicitRepresentation(
        volume_shape=stack.shape,
        constant_init_values=params.init_values,
    )
    render_model = Render_engine(
        image_size_polar=dataset.sh.frame_size_cart_pix,
        constant_init_values=params.init_values,
        compression=params.compression,
    )

    loss, l2, ssim = train_model(
        representation_model=representation_model,
        pose_model=pose_model,
        render_model=render_model,
        data=dataset,
        training_params=params,
        device=device,
        progress_callback=progress_callback,
    )

    if not silent:
        visualize_stats(loss, l2, ssim, output_dir=output_dir.joinpath("training_stats"))

    shadow_reduced, _ = render_volume(
        representation_model=representation_model,
        pose_model=pose_model,
        render_model=render_model,
        dataset=dataset,
        batch_size=params.params_generator["batch_size"],
    )

    if resample_output:
        if geometry is None:
            raise ValueError("geometry is required when resample_output=True.")
        shadow_reduced = resample_to_image_space(shadow_reduced, geometry, params)

    shadow_reduced_path = output_dir.joinpath("shadow_removed", "shadow_reduced_volume.nii.gz")
    original_path = output_dir.joinpath("shadow_removed", "original_volume.nii.gz")
    save_volume(
        volume=to_8bit_graysacle(shadow_reduced, volume_mask=volume_mask),
        file_path=shadow_reduced_path,
        header=get_medpy_header(),
    )
    save_volume(
        volume=to_8bit_graysacle(original_stack, volume_mask=volume_mask),
        file_path=original_path,
        header=get_medpy_header(),
    )
    return RFlashOutput(
        shadow_reduced_path=shadow_reduced_path,
        original_path=original_path,
        output_dir=output_dir,
        shadow_reduced_data=shadow_reduced,
        original_data=original_stack,
        header=get_medpy_header(),
        spacing_mm=np.asarray([1.0, 1.0, 1.0]),
        is_volume=False,
    )


def _apply_training_overrides(
    params: Parameter_Demo3D | Parameter_Demo2D_curvylinear | Parameter_Demo2D_linear,
    training_overrides: dict[str, float | int] | None,
) -> None:
    """Apply validated training overrides to a demo parameter object."""

    if training_overrides is None:
        return

    if "max_epochs" in training_overrides:
        params.max_epochs = int(training_overrides["max_epochs"])
    if "lr" in training_overrides:
        params.lr = float(training_overrides["lr"])
    if "lamda_train" in training_overrides:
        params.lamda_train = float(training_overrides["lamda_train"])
    if "compression" in training_overrides:
        params.compression = float(training_overrides["compression"])
    if "lr_shed_patience" in training_overrides:
        params.lr_shed_patience = int(training_overrides["lr_shed_patience"])
    if "batch_size" in training_overrides:
        params.params_generator["batch_size"] = int(training_overrides["batch_size"])
