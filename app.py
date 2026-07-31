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
    Gradio web interface for running the public RFlash shadow-reduction demo on
    uploaded or server-side ultrasound data.
License:
    This file is part of the RFlash project and is distributed under the
    repository's LICENSE. See the LICENSE file in the repository root for
    licensing information.
------------------------------------------------------------------------------"""

from __future__ import annotations

import argparse
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np

from src.inference import (
    RFlashOutput,
    run_3d_volume_inference,
    run_curvilinear_stack_inference,
    run_linear_stack_inference,
    select_device,
)
from src.utils.io import load_image_directory, load_image_files, load_volume, normalize_for_display
from src.utils.transducer_geometry import (
    estimate_geometry_slice,
    estimate_scanner_geometry_volume,
)

INPUT_VOLUME = "3D ultrasound volume (.mha, .nii, .nii.gz)"
INPUT_STACK = "2D image stack or single 2D image"
PROBE_LINEAR = "linear"
PROBE_CURVILINEAR = "curvilinear"
GEOMETRY_SHARED = "all images share one scanner geometry"
GEOMETRY_PER_IMAGE = "estimate geometry for each image"


@dataclass
class GeometryState:
    """Geometry estimates prepared by the web interface.

    Attributes:
        input_kind: User-selected input kind.
        probe: User-selected probe type.
        geometry_mode: Geometry sharing mode for curvilinear stacks.
        geometry: Estimated scanner geometry object or per-slice geometries.
        indices: Stack indices associated with per-slice geometries.
    """

    input_kind: str
    probe: str | None
    geometry_mode: str | None
    geometry: Any
    indices: list[int] | None = None


def build_interface(default_output_dir: Path) -> gr.Blocks:
    """Create the Gradio Blocks interface.

    Args:
        default_output_dir: Directory used when the user leaves the output
            directory field empty.

    Returns:
        Configured Gradio interface.
    """

    with gr.Blocks(title="RFlash") as app:
        gr.Markdown("# RFlash")
        gr.Markdown("Run ultrasound shadow reduction from a volume, a 2D image stack, or a single 2D image.")

        geometry_state = gr.State(None)

        with gr.Row():
            with gr.Column(scale=1):
                uploaded_files = gr.File(
                    label="Upload data",
                    file_count="multiple",
                    type="filepath",
                    file_types=[".mha", ".nii", ".gz", ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".npy"],
                )
                server_path = gr.Textbox(
                    label="Or use a server-side path",
                    placeholder="/path/to/volume.mha, /path/to/images, or /path/to/images.npy",
                )
                input_kind = gr.Radio(
                    choices=[INPUT_VOLUME, INPUT_STACK],
                    value=INPUT_VOLUME,
                    label="Uploaded data type",
                )
                probe = gr.Radio(
                    choices=[PROBE_LINEAR, PROBE_CURVILINEAR],
                    value=PROBE_CURVILINEAR,
                    label="Probe type",
                    visible=False,
                )
                geometry_mode = gr.Radio(
                    choices=[GEOMETRY_SHARED, GEOMETRY_PER_IMAGE],
                    value=GEOMETRY_SHARED,
                    label="Scanner geometry",
                    visible=False,
                )
                num_slices = gr.Number(
                    label="Number of slices to process",
                    value=1,
                    precision=0,
                    minimum=1,
                    visible=False,
                )
                output_dir = gr.Textbox(
                    label="Output directory",
                    value=str(default_output_dir),
                )
                confirm_geometry = gr.Checkbox(
                    label="I confirm the displayed geometry",
                    value=False,
                    visible=True,
                )
                with gr.Row():
                    preview_button = gr.Button("Estimate geometry", variant="secondary")
                    run_button = gr.Button("Run RFlash", variant="primary")

            with gr.Column(scale=2):
                status = gr.Textbox(label="Status", interactive=False)
                geometry_gallery = gr.Gallery(label="Estimated geometry", columns=2, height=360)
                output_gallery = gr.Gallery(label="Representative outputs", columns=2, height=420)
                download_output = gr.File(label="Download processed output")

        input_kind.change(
            _update_visible_controls,
            inputs=[input_kind, probe],
            outputs=[probe, geometry_mode, num_slices, confirm_geometry, preview_button],
        )
        probe.change(
            _update_visible_controls,
            inputs=[input_kind, probe],
            outputs=[probe, geometry_mode, num_slices, confirm_geometry, preview_button],
        )
        preview_button.click(
            _estimate_geometry_for_interface,
            inputs=[uploaded_files, server_path, input_kind, probe, geometry_mode, num_slices, output_dir],
            outputs=[geometry_state, geometry_gallery, status, confirm_geometry],
        )
        run_button.click(
            _run_rflash_for_interface,
            inputs=[
                uploaded_files,
                server_path,
                input_kind,
                probe,
                geometry_mode,
                num_slices,
                output_dir,
                confirm_geometry,
                geometry_state,
            ],
            outputs=[output_gallery, download_output, status],
        )

    return app


def _update_visible_controls(input_kind: str, probe: str) -> tuple:
    """Show only controls relevant to the selected workflow branch."""

    stack_selected = input_kind == INPUT_STACK
    curvilinear_selected = stack_selected and probe == PROBE_CURVILINEAR
    geometry_required = input_kind == INPUT_VOLUME or curvilinear_selected
    return (
        gr.update(visible=stack_selected),
        gr.update(visible=curvilinear_selected),
        gr.update(visible=curvilinear_selected),
        gr.update(visible=geometry_required, value=False),
        gr.update(visible=geometry_required),
    )


def _estimate_geometry_for_interface(
    uploaded_files: list[str] | None,
    server_path: str,
    input_kind: str,
    probe: str,
    geometry_mode: str,
    num_slices: float,
    output_dir: str,
) -> tuple[GeometryState | None, list[str], str, gr.Checkbox]:
    """Estimate scanner geometry and save overlays for user confirmation."""

    resolved_input = _resolve_input(uploaded_files, server_path)
    output_root = _resolve_output_dir(output_dir)
    overlay_dir = output_root / "intermediate_images"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    if input_kind == INPUT_VOLUME:
        volume, header = load_volume(resolved_input)
        spacing_mm = np.asarray([float(value) for value in header.spacing])
        geometry = estimate_scanner_geometry_volume(volume, spacing_mm, overlay_dir=overlay_dir, verbose=False)
        overlays = _find_geometry_overlays(overlay_dir)
        state = GeometryState(input_kind=input_kind, probe=None, geometry_mode=None, geometry=geometry)
        return state, overlays, "Estimated 3D scanner geometry. Please inspect the overlays before running.", gr.update(value=False)

    if probe != PROBE_CURVILINEAR:
        return None, [], "Linear-probe inputs do not require scanner geometry estimation.", gr.update(value=True)

    stack = _load_web_image_stack(resolved_input)
    requested_slices = _validate_num_slices(num_slices, stack.shape[0])
    if geometry_mode == GEOMETRY_SHARED:
        middle_idx = stack.shape[0] // 2
        overlay_path = overlay_dir / "geometry_shared.png"
        geometry, success = estimate_geometry_slice(stack[middle_idx], overlay_path=overlay_path, verbose=False)
        if not success or geometry is None:
            raise gr.Error("Scanner geometry estimation failed for the shared geometry slice.")
        selected_indices = list(range(min(requested_slices, stack.shape[0])))
        geometries = [geometry for _ in selected_indices]
        overlays = [str(overlay_path)]
        state = GeometryState(
            input_kind=input_kind,
            probe=probe,
            geometry_mode=geometry_mode,
            geometry=geometries,
            indices=selected_indices,
        )
    else:
        geometries, selected_indices, overlays = _estimate_per_slice_geometries_for_interface(
            stack,
            spacing_mm=(1.0, 1.0),
            overlay_dir=overlay_dir,
            num_slices=requested_slices,
        )
        state = GeometryState(
            input_kind=input_kind,
            probe=probe,
            geometry_mode=geometry_mode,
            geometry=geometries,
            indices=selected_indices,
        )

    return state, overlays, "Estimated scanner geometry. Please inspect the overlays before running.", gr.update(value=False)


def _run_rflash_for_interface(
    uploaded_files: list[str] | None,
    server_path: str,
    input_kind: str,
    probe: str,
    geometry_mode: str,
    num_slices: float,
    output_dir: str,
    confirm_geometry: bool,
    geometry_state: GeometryState | None,
) -> tuple[list[str], str | None, str]:
    """Run RFlash from the current interface selections."""

    resolved_input = _resolve_input(uploaded_files, server_path)
    output_root = _resolve_output_dir(output_dir)
    device = select_device(prefer_cuda=True)

    if input_kind == INPUT_VOLUME:
        if geometry_state is None or not confirm_geometry:
            raise gr.Error("Please estimate and confirm the scanner geometry before running RFlash.")
        result = run_3d_volume_inference(
            input_path=resolved_input,
            output_dir=output_root,
            device=device,
            silent=True,
            geometry=geometry_state.geometry,
        )
    elif probe == PROBE_CURVILINEAR:
        if geometry_state is None or not confirm_geometry:
            raise gr.Error("Please estimate and confirm the scanner geometry before running RFlash.")
        result = run_curvilinear_stack_inference(
            input_path=resolved_input,
            output_dir=output_root,
            device=device,
            silent=True,
            geometry=geometry_state.geometry,
            indices=geometry_state.indices,
            num_slices=_validate_num_slices(num_slices, _infer_stack_length(resolved_input)),
        )
    else:
        result = run_linear_stack_inference(
            input_path=resolved_input,
            output_dir=output_root,
            device=device,
            silent=True,
        )

    previews = _write_output_previews(result)
    status = f"RFlash finished on {device}. Output written to {result.shadow_reduced_path}."
    return previews, str(result.shadow_reduced_path), status


def _resolve_input(uploaded_files: list[str] | None, server_path: str) -> str | Path | list[str | Path]:
    """Resolve uploaded files or a server-side path to an inference input."""

    if server_path.strip():
        return Path(server_path.strip()).expanduser()

    files = _as_path_list(uploaded_files)
    if not files:
        raise gr.Error("Please upload data or enter a server-side path.")
    if len(files) == 1:
        return files[0]
    return files


def _resolve_output_dir(output_dir: str) -> Path:
    """Resolve the requested output directory."""

    if output_dir.strip():
        return Path(output_dir.strip()).expanduser()
    return Path(tempfile.mkdtemp(prefix="rflash-gradio-"))


def _as_path_list(uploaded_files: list[Any] | None) -> list[Path]:
    """Convert Gradio file values to paths."""

    if uploaded_files is None:
        return []
    if isinstance(uploaded_files, (str, Path)):
        uploaded_files = [uploaded_files]
    paths = []
    for file_value in uploaded_files:
        file_path = getattr(file_value, "name", file_value)
        paths.append(Path(file_path))
    return paths


def _load_web_image_stack(input_value: str | Path | list[str | Path]) -> np.ndarray:
    """Load 2D image uploads or image directories for geometry preview."""

    if isinstance(input_value, list):
        return load_image_files(input_value)
    path = Path(input_value)
    if path.is_dir():
        return load_image_directory(path)
    return load_image_files([path])


def _infer_stack_length(input_value: str | Path | list[str | Path]) -> int:
    """Return the number of 2D images in the selected stack."""

    return _load_web_image_stack(input_value).shape[0]


def _validate_num_slices(num_slices: float, max_slices: int) -> int:
    """Validate a Gradio numeric slice count."""

    try:
        value = int(num_slices)
    except (TypeError, ValueError) as exc:
        raise gr.Error("Number of slices must be an integer.") from exc
    if value < 1 or value > max_slices:
        raise gr.Error(f"Number of slices must be between 1 and {max_slices}.")
    return value


def _find_geometry_overlays(overlay_dir: Path) -> list[str]:
    """Find saved geometry overlay images for display."""

    return [str(path) for path in sorted(overlay_dir.glob("geometry*.png"))]


def _estimate_per_slice_geometries_for_interface(
    stack: np.ndarray,
    spacing_mm: tuple[float, float],
    overlay_dir: Path,
    num_slices: int,
) -> tuple[list[Any], list[int], list[str]]:
    """Estimate per-slice geometries with stable overlay filenames."""

    geometries = []
    selected_indices = []
    overlays = []
    for slice_idx in range(num_slices):
        overlay_path = overlay_dir / f"geometry_slice_{slice_idx:03d}.png"
        geometry, success = estimate_geometry_slice(
            stack[slice_idx],
            spacing_plane_mm=spacing_mm,
            overlay_path=overlay_path,
            verbose=False,
        )
        if not success or geometry is None:
            raise gr.Error(f"Scanner geometry estimation failed for slice {slice_idx}.")
        geometries.append(geometry)
        selected_indices.append(slice_idx)
        overlays.append(str(overlay_path))
    return geometries, selected_indices, overlays


def _write_output_previews(result: RFlashOutput) -> list[str]:
    """Write representative original/processed slice previews."""

    original, _ = load_volume(result.original_path)
    processed, _ = load_volume(result.shadow_reduced_path)
    preview_dir = result.output_dir / "web_previews"
    preview_dir.mkdir(parents=True, exist_ok=True)

    preview_paths = []
    for label, volume in (("original", original), ("shadow_reduced", processed)):
        for slice_idx in _representative_slice_indices(volume):
            image = _slice_for_display(volume, slice_idx)
            output_path = preview_dir / f"{label}_slice_{slice_idx:03d}.png"
            _save_preview_image(image, output_path, f"{label.replace('_', ' ')} slice {slice_idx}")
            preview_paths.append(str(output_path))
    return preview_paths


def _representative_slice_indices(volume: np.ndarray) -> list[int]:
    """Choose a small set of informative slice indices for display."""

    if volume.ndim != 3:
        return [0]
    last_axis = volume.shape[-1]
    if last_axis == 1:
        return [0]
    return sorted(set([last_axis // 4, last_axis // 2, (3 * last_axis) // 4]))


def _slice_for_display(volume: np.ndarray, slice_idx: int) -> np.ndarray:
    """Extract one display slice from a saved volume."""

    if volume.ndim == 2:
        return volume
    return volume[..., slice_idx]


def _save_preview_image(image: np.ndarray, output_path: Path, title: str) -> None:
    """Save one grayscale preview image."""

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(normalize_for_display(image), cmap="gray")
    ax.set_title(title)
    ax.axis("off")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _env_share_default() -> bool:
    """Read the default Gradio sharing setting from ``RFLASH_GRADIO_SHARE``."""

    return os.environ.get("RFLASH_GRADIO_SHARE", "").strip().lower() in {"1", "true", "yes", "on"}


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for the Gradio app."""

    parser = argparse.ArgumentParser(description="Run the RFlash Gradio web interface.")
    parser.add_argument("--share", action="store_true", default=_env_share_default(), help="Enable Gradio sharing.")
    parser.add_argument("--no-share", action="store_false", dest="share", help="Disable Gradio sharing.")
    parser.add_argument("--server-port", type=int, default=int(os.environ.get("PORT", 7860)), help="Server port.")
    parser.add_argument("--server-name", default="0.0.0.0", help="Server host. Defaults to 0.0.0.0.")
    parser.add_argument("--output", type=Path, default=Path("outputs/gradio"), help="Default output directory.")
    return parser.parse_args()


def main() -> None:
    """Launch the Gradio web interface."""

    args = parse_args()
    app = build_interface(default_output_dir=args.output)
    app.launch(server_name=args.server_name, server_port=args.server_port, share=args.share)


if __name__ == "__main__":
    main()
