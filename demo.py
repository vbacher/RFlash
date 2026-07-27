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
    Command line entry point for running the public RFlash shadow-reduction
    demo on fetal brain, abdominal ultrasound, or synthetic liver data.
License:
    This file is part of the RFlash project and is distributed under the
    repository's LICENSE. See the LICENSE file in the repository root for
    licensing information.
------------------------------------------------------------------------------"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

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


def run_3D_volume_demo(args: argparse.Namespace, device: torch.device) -> str:
    """Run the full RFlash demo for a 3D ultrasound volume.

    Args:
        args: Parsed command line arguments. ``args.input`` must point to a
            supported 3D volume file, and ``args.output`` selects the output
            directory.
        device: PyTorch device used for model training and rendering.

    Returns:
        The path to the shadow-reduced volume.
    """

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    volume, header = load_volume(args.input)
    spacing_mm = np.asarray([float(value) for value in header.spacing])

    # Geometry estimation is done from the observed fan support before
    # standardization so that background zeros remain meaningful.
    geometry = estimate_scanner_geometry_volume(
        volume, spacing_mm, overlay_dir=f"{output_dir}/intermediate_images", verbose=not args.silent
    )
    params = Parameter_Demo3D()
    params.image_size_polar = (np.asarray(volume.shape[::-2]) * 0.9).astype(np.int16)
    params.image_size_cartesian = (np.asarray(volume.shape[::-2]) * 0.9).astype(np.int16)
    params.num_fan_slices = int(volume.shape[1] * 0.9)

    volume = standardize_volume(volume, params.init_values[1])

    # Create training dataset.
    dataset = Dataset_3D_volume(
        volume=volume, model_params=params, geometry=geometry, pixel_spacing_mm=spacing_mm, device=device
    )

    # Initialize the fixed pose model, the trainable explicit parameter volume,
    # and the differentiable ultrasound renderer.
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

    # Train the decomposition model while logging scalar statistics in memory.
    loss, l2, ssim = train_model(
        representation_model=representation_model,
        pose_model=pose_model,
        render_model=render_model,
        data=dataset,
        training_params=params,
        device=device,
    )
    if not args.silent:
        visualize_stats(loss, l2, ssim, output_dir=output_dir.joinpath("training_stats"))

    # Re-render the learned representation without the attenuation term to
    # obtain the shadow-reduced volume.
    shadow_reduced, _ = render_volume(
        representation_model=representation_model,
        pose_model=pose_model,
        render_model=render_model,
        dataset=dataset,
        batch_size=params.params_generator["batch_size"],
    )

    o_path_shadow_reduced = output_dir.joinpath("shadow_removed", "shadow_reduced_volume.nii.gz")
    save_volume(
        volume=to_8bit_graysacle(shadow_reduced, volume_mask=shadow_reduced > 0.01),
        file_path=o_path_shadow_reduced,
        header=header,
    )
    save_volume(
        volume=to_8bit_graysacle(volume, volume_mask=volume > 0.01),
        file_path=output_dir.joinpath("shadow_removed", "original_volume.nii.gz"),
        header=header,
    )

    return o_path_shadow_reduced


def run_2D_stack_demo(args: argparse.Namespace, device: torch.device) -> str:
    """Run the RFlash demo for a stack of curvilinear 2D ultrasound images.

    Args:
        args: Parsed command line arguments. ``args.input`` must be a directory
            of grayscale-compatible image files.
        device: PyTorch device used for model training and rendering.

    Returns:
        The path to the shadow-reduced volume.
    """

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    stack = load_image_directory(args.input)
    geometry, indices = estimate_scanner_geometries_stack(
        stack, spacing_mm=(1.0, 1.0), overlay_dir=f"{output_dir}/intermediate_images", verbose=not args.silent
    )
    stack = stack[indices]
    stack = np.transpose(stack, (1, 2, 0))

    params = Parameter_Demo2D_curvylinear()
    params.image_size_polar = (np.asarray(stack.shape[:-1]) * 1.1).astype(np.int16)
    params.image_size_cartesian = np.asarray(stack.shape[:-1]).astype(np.int16)

    # Curvilinear 2D slices are resampled into the renderer's polar simulation
    # space, where they can be optimized using the same linear-stack machinery.
    stack_sim = resample_to_simulation_space(stack, geometry, params)

    vol_mask = resample_to_image_space(np.ones_like(stack, dtype=bool), geometry, params).astype(bool)

    stack_sim = standardize_volume(stack_sim, params.init_values[1])

    # Create training dataset.
    dataset = Dataset_2D_lin_array(stack=stack_sim)

    # Initialize the fixed pose model, the trainable explicit parameter stack,
    # and the differentiable ultrasound renderer.
    pose_model = SlicePoses(dataset.get_localization())
    representation_model = ExplicitRepresentation(
        volume_shape=stack_sim.shape,
        constant_init_values=params.init_values,
    )
    render_model = Render_engine(
        image_size_polar=dataset.sh.frame_size_cart_pix,
        constant_init_values=params.init_values,
        compression=params.compression,
    )

    # Train the decomposition model while logging scalar statistics in memory.
    loss, l2, ssim = train_model(
        representation_model=representation_model,
        pose_model=pose_model,
        render_model=render_model,
        data=dataset,
        training_params=params,
        device=device,
    )

    if not args.silent:
        visualize_stats(loss, l2, ssim, output_dir=output_dir.joinpath("training_stats"))

    # Re-render without attenuation, then map back to the original image space
    # so the saved output aligns with the input images.
    shadow_reduced, _ = render_volume(
        representation_model=representation_model,
        pose_model=pose_model,
        render_model=render_model,
        dataset=dataset,
        batch_size=params.params_generator["batch_size"],
    )

    shadow_reduced = resample_to_image_space(shadow_reduced, geometry, params)

    o_path_shadow_reduced = output_dir.joinpath("shadow_removed", "shadow_reduced_volume.nii.gz")
    save_volume(
        volume=to_8bit_graysacle(shadow_reduced, volume_mask=vol_mask),
        file_path=output_dir.joinpath("shadow_removed", "shadow_reduced_volume.nii.gz"),
        header=get_medpy_header(),
    )
    save_volume(
        volume=to_8bit_graysacle(stack, volume_mask=vol_mask),
        file_path=output_dir.joinpath("shadow_removed", "original_volume.nii.gz"),
        header=get_medpy_header(),
    )
    return o_path_shadow_reduced


def run_synthetic_liver_demo(args: argparse.Namespace, device: torch.device) -> str:
    """Run the RFlash demo for synthetic linear-probe liver ultrasound data.

    Args:
        args: Parsed command line arguments. ``args.input`` can be a single
            ``.npy`` file or a directory of compatible ``.npy`` files.
        device: PyTorch device used for model training and rendering.

    Returns:
        The path to the shadow-reduced volume.
    """

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    stack = load_synthetic_liver(args.input)

    params = Parameter_Demo2D_linear()
    stack = standardize_volume(stack, params.init_values[1])

    # Create training dataset.
    dataset = Dataset_2D_lin_array(stack=stack)

    # Initialize the fixed pose model, the trainable explicit parameter stack,
    # and the differentiable ultrasound renderer.
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

    # Train the decomposition model while logging scalar statistics in memory.
    loss, l2, ssim = train_model(
        representation_model=representation_model,
        pose_model=pose_model,
        render_model=render_model,
        data=dataset,
        training_params=params,
        device=device,
    )
    if not args.silent:
        visualize_stats(loss, l2, ssim, output_dir=output_dir.joinpath("training_stats"))

    # The synthetic liver input is already in the renderer's linear simulation
    # space, so the learned shadow-reduced stack can be saved directly.
    shadow_reduced, _ = render_volume(
        representation_model=representation_model,
        pose_model=pose_model,
        render_model=render_model,
        dataset=dataset,
        batch_size=params.params_generator["batch_size"],
    )

    o_path_shadow_reduced = output_dir.joinpath("shadow_removed", "shadow_reduced_volume.nii.gz")
    save_volume(
        volume=to_8bit_graysacle(shadow_reduced, volume_mask=np.ones_like(shadow_reduced, dtype=bool)),
        file_path=o_path_shadow_reduced,
        header=get_medpy_header(),
    )
    save_volume(
        volume=to_8bit_graysacle(stack, volume_mask=np.ones_like(stack, dtype=bool)),
        file_path=output_dir.joinpath("shadow_removed", "original_volume.nii.gz"),
        header=get_medpy_header(),
    )
    return o_path_shadow_reduced


def run_demo(args: argparse.Namespace) -> None:
    """Dispatch the command line demo for the selected dataset.

    Args:
        args: Parsed command line arguments from :func:`parse_args`.

    Returns:
        None.

    Raises:
        ValueError: If ``args.dataset`` is not one of ``DATASET_CHOICES``.
    """

    # Prefer accelerator backends when available; all tensors are moved inside
    # the training routine and dataset constructors.
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
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

    print(f"\n Demo finished. Outputs written to {o_path}.")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for the public demo.

    Returns:
        Parsed command line arguments.
    """

    parser = argparse.ArgumentParser(
        description="Load ultrasound demo data and estimate the RFlash scanner geometry.",
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=Path("data/fetal_brain/test_3d.nii.gz"),
        help="Input .mha file, image directory, .npy file, or .npy directory.",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        choices=DATASET_CHOICES,
        default="fetal_brain",
        help="Dataset format to load.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("outputs"),
        help="Directory for intermediate images, training plots, and saved volumes.",
    )
    parser.add_argument(
        "-s",
        "--silent",
        dest="silent",
        action="store_true",
        help="silence intermediate output and non-essential user interactions",
    )
    parser.set_defaults(confirm_geometry=sys.stdin.isatty(), show_overlay=sys.stdin.isatty())
    return parser.parse_args()


if __name__ == "__main__":
    run_demo(parse_args())
