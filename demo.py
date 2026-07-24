"""Command line demo for loading RFlash example data and estimating geometry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from networkx import volume
import numpy as np
import torch

from src.utils.transducer_geometry import estimate_scanner_geometry_volume, estimate_scanner_geometries_stack
from src.utils.io import (
    load_image_directory,
    load_synthetic_liver,
    load_volume,
    save_volume,
    to_8bit_graysacle,
    get_medpy_header,
)
from src.utils.transformations import standardize_volume
from src.datasets import Dataset_3D_volume, Dataset_2D_lin_array
from src.model.representation import SlicePoses, ExplicitRepresentation
from src.model.rendering import Render_engine
from src.shadow_reduction import render_volume

from src.trainings_params import Parameter_Demo3D, Parameter_Demo2D
from src.train import train_model

from src.utils.visualization import visualize_stats

# FIXME:remove
from src.utils.visualization import visualize_2d_image


DATASET_CHOICES = ("fetal_brain", "abdominal", "synthetic_liver")


def run_3D_volume_demo(args: argparse.Namespace, device: torch.device) -> None:

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    volume, header = load_volume(args.input)
    spacing_mm = np.asarray([float(value) for value in header.spacing])

    # estimate transducer geometry
    geometry = estimate_scanner_geometry_volume(
        volume, spacing_mm, overlay_dir=f"{output_dir}/intermediate_images", verbose=not args.silent
    )
    params = Parameter_Demo3D()
    params.image_size_polar = (np.asarray(volume.shape[::-2]) * 0.9).astype(np.int16)
    params.image_size_cartesian = (np.asarray(volume.shape[::-2]) * 0.9).astype(np.int16)
    params.num_fan_slices = int(volume.shape[1] * 0.9)

    volume = standardize_volume(volume, params.init_values[1])

    # create traiings dataset
    dataset = Dataset_3D_volume(
        volume=volume, model_params=params, geometry=geometry, pixel_spacing_mm=spacing_mm, device=device
    )

    # initialize models
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

    # train decomposition model
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

    # Obtain shadow reduced volume from the trained model
    shadow_reduced, _ = render_volume(
        representation_model=representation_model,
        pose_model=pose_model,
        render_model=render_model,
        dataset=dataset,
        batch_size=params.params_generator["batch_size"],
    )

    save_volume(
        volume=to_8bit_graysacle(shadow_reduced, volume_mask=shadow_reduced > 0.01),
        file_path=output_dir.joinpath("shadow_removed", "shadow_reduced_volume.nii.gz"),
        header=header,
    )
    save_volume(
        volume=to_8bit_graysacle(volume, volume_mask=volume > 0.01),
        file_path=output_dir.joinpath("shadow_removed", "original_volume.nii.gz"),
        header=header,
    )


def run_2D_stack_demo(args: argparse.Namespace, device: torch.device) -> None:

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    stack = load_image_directory(args.input)
    geometry, indices = estimate_scanner_geometries_stack(
        stack, spacing_mm=(1.0, 1.0), overlay_dir=f"{output_dir}/intermediate_images", verbose=not args.silent
    )
    stack = stack[indices]


def run_synthetic_liver_demo(args: argparse.Namespace, device: torch.device) -> None:
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    stack = load_synthetic_liver(args.input)

    params = Parameter_Demo2D()
    stack = standardize_volume(stack, params.init_values[1])

    # create traiings dataset
    dataset = Dataset_2D_lin_array(stack=stack)

    # initialize models
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

    # train decomposition model
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

    # Obtain shadow reduced volume from the trained model
    shadow_reduced, _ = render_volume(
        representation_model=representation_model,
        pose_model=pose_model,
        render_model=render_model,
        dataset=dataset,
        batch_size=params.params_generator["batch_size"],
    )

    save_volume(
        volume=to_8bit_graysacle(shadow_reduced, volume_mask=np.ones_like(shadow_reduced, dtype=bool)),
        file_path=output_dir.joinpath("shadow_removed", "shadow_reduced_volume.nii.gz"),
        header=get_medpy_header(),
    )
    save_volume(
        volume=to_8bit_graysacle(stack, volume_mask=np.ones_like(stack, dtype=bool)),
        file_path=output_dir.joinpath("shadow_removed", "original_volume.nii.gz"),
        header=get_medpy_header(),
    )


def run_demo(args: argparse.Namespace) -> None:
    """Load a dataset and estimate its ultrasound fan geometry."""

    # get execution device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    if not args.silent:
        print(f"executing on {device} device")

    if args.dataset == "fetal_brain":
        run_3D_volume_demo(args, device=device)

    elif args.dataset == "abdominal":
        run_2D_stack_demo(args, device=device)

    elif args.dataset == "synthetic_liver":
        run_synthetic_liver_demo(args, device=device)

    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for the public demo."""

    parser = argparse.ArgumentParser(
        description="Load ultrasound demo data and estimate the RFlash scanner geometry.",
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        # default=Path("data/fetal_brain/test_3d.nii.gz"),
        # default="/home/scratch/valher/data/RFlash-demo/archive/abdominal_US/abdominal_US/RUS/images/train"
        default=Path("/home/scratch/valher/data/RFlash-demo/syn_liver"),
        help="Input .mha file, image directory, .npy file, or .npy directory.",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        choices=DATASET_CHOICES,
        # default="fetal_brain",
        default="synthetic_liver",
        help="Dataset format to load.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("outputs"),
        help="Directory for geometry overlays and geometry.json.",
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
