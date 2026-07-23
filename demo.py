"""Command line demo for loading RFlash example data and estimating geometry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

from src.transducer_geometry import estimate_scanner_geometry_volume, estimate_scanner_geometries_stack
from src.utils import load_image_directory, load_synthetic_liver, load_volume, to_jsonable

# FIXME:remove
from src.utils import visualize_2d_image


DATASET_CHOICES = ("fetal_brain", "abdominal", "synthetic_liver")


def run_demo(args: argparse.Namespace) -> dict:
    """Load a dataset and estimate its ultrasound fan geometry."""

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset == "fetal_brain":
        volume, header = load_volume(args.input)
        spacing_mm = tuple(float(value) for value in header.spacing)
        geometry = estimate_scanner_geometry_volume(
            volume, spacing_mm, overlay_dir=f"{output_dir}/intermediate_images", verbose=not args.silent
        )
        print("continue implementing")
    elif args.dataset == "abdominal":
        stack = load_image_directory(args.input)
        geometry, indices = estimate_scanner_geometries_stack(
            stack, spacing_mm=(1.0, 1.0), overlay_dir=f"{output_dir}/intermediate_images", verbose=not args.silent
        )
        stack = stack[indices]
        print("continue implementing")
    elif args.dataset == "synthetic_liver":
        stack = load_synthetic_liver(args.input)
        print("continue implementing")

    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    return {}


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
        default="/home/scratch/valher/data/RFlash-demo/archive/abdominal_US/abdominal_US/RUS/images/train",
        help="Input .mha file, image directory, .npy file, or .npy directory.",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        choices=DATASET_CHOICES,
        # default="fetal_brain",
        default="abdominal",
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
    result = run_demo(parse_args())
    print(json.dumps(to_jsonable(result), indent=2))
