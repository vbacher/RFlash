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

from src.inference import (
    DATASET_CHOICES,
    run_2D_stack_demo,
    run_3D_volume_demo,
    run_demo,
    run_synthetic_liver_demo,
)


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
        default=Path("data/fetal_brain/fetal-brain-demo.mha"),
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
        "--output-mha",
        type=Path,
        help="Optional exact .mha path for a copy of the shadow-reduced output.",
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
