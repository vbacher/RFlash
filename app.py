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
import queue
import shutil
import tempfile
import threading
import traceback
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from src.inference import (
    RFlashOutput,
    run_3d_volume_inference,
    run_curvilinear_stack_inference,
    run_linear_stack_inference,
    select_device,
)
from src.utils.io import (
    get_medpy_header,
    load_image_directory,
    load_image_files,
    load_synthetic_liver,
    load_volume,
    normalize_for_display,
    save_volume,
    to_8bit_graysacle,
)
from src.utils.transducer_geometry import (
    estimate_geometry_slice,
    estimate_scanner_geometry_volume,
)
from src.utils.visualization import visualize_stats

INPUT_VOLUME = "3D ultrasound volume"
INPUT_STACK = "2D image stack or single 2D image"
PROBE_LINEAR = "linear"
PROBE_CURVILINEAR = "curvilinear"
GEOMETRY_SHARED = "all images share one scanner geometry"
GEOMETRY_PER_IMAGE = "estimate geometry for each image"

FORMAT_NII = "nii.gz"
FORMAT_MHA = "mha"
FORMAT_ZIP_PNG = "zip with PNG slices"
FORMAT_ZIP_JPG = "zip with JPG slices"
FORMAT_MP4 = "mp4 video"


@dataclass
class GeometryWorkflowState:
    """Interactive geometry-selection state for stack-based workflows."""

    input_kind: str
    probe: str | None
    geometry_mode: str | None
    target_count: int
    stack_data: np.ndarray | None = None
    stack_shape: tuple[int, int, int] | None = None
    overlay_paths: list[str] = field(default_factory=list)
    candidate_indices: list[int] = field(default_factory=list)
    accepted_indices: list[int] = field(default_factory=list)
    accepted_geometries: list[Any] = field(default_factory=list)
    accepted_overlays: list[str] = field(default_factory=list)
    current_candidate_position: int = 0
    current_candidate_index: int | None = None
    current_geometry: Any | None = None
    current_overlay: str | None = None
    completed: bool = False
    volume_geometry: Any | None = None
    volume_data: np.ndarray | None = None
    volume_spacing_mm: np.ndarray | None = None
    output_dir: Path | None = None


def build_interface(default_output_dir: Path) -> gr.Blocks:
    """Create the Gradio Blocks interface."""

    default_input = Path(__file__).resolve().parent / "data" / "fetal_brain" / "fetal-brain-demo.mha"
    # ``file_count="multiple"`` requires a list-shaped default in Gradio.
    default_input_value = [str(default_input)] if default_input.exists() else None

    with gr.Blocks(title="RFlash") as app:
        gr.Markdown("# RFlash")
        gr.Markdown("Run RFlash on a 3D ultrasound volume, a stack of 2D images, or a single 2D image.")
        gr.Markdown(
            "Upload your own data or use the "
            "[fetal brain](https://github.com/vbacher/RFlash/tree/main/data/fetal_brain), "
            "[abdominal ultrasound](https://www.kaggle.com/datasets/ignaciorlando/ussimandsegm), or "
            "[synthetic liver](https://github.com/magdalena-wysocki/ultra-nerf/tree/main/data/synthetic_testing) "
            "datasets. For more information on the data please refer to the "
            "[GitHub Repo](https://github.com/vbacher/RFlash/blob/main/README.md)."
        )

        geometry_state = gr.State(None)

        with gr.Row():
            with gr.Column(scale=1):
                uploaded_files = gr.File(
                    label="Upload data",
                    file_count="multiple",
                    value=default_input_value,
                    type="filepath",
                    file_types=[".mha", ".nii", ".gz", ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".npy"],
                    height=180,
                )

                input_kind = gr.Radio(
                    choices=[INPUT_VOLUME, INPUT_STACK],
                    value=INPUT_VOLUME,
                    label="Input type",
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
                    label="Curvilinear geometry handling",
                    visible=False,
                )
                num_slices = gr.Number(
                    label="Number of slices to process",
                    value=1,
                    precision=0,
                    minimum=1,
                    visible=False,
                )
                output_format = gr.Radio(
                    choices=[FORMAT_NII, FORMAT_MHA],
                    value=FORMAT_NII,
                    label="Processed output format",
                )

                accepted_types = gr.Markdown(value=_accepted_data_message(INPUT_VOLUME, PROBE_CURVILINEAR))

                advanced = gr.Checkbox(
                    label="Advanced settings",
                    value=False,
                )

                with gr.Group(visible=False) as advanced_settings:
                    server_path = gr.Textbox(
                        label="Server-side input path",
                        placeholder="/path/to/volume.mha, /path/to/images, /path/to/stack.nii.gz, or /path/to/images.npy",
                    )
                    output_dir = gr.Textbox(
                        label="Output directory",
                        value="" if default_output_dir == Path("outputs/gradio") else str(default_output_dir),
                        placeholder="Optional output directory (a new temporary folder is used when empty)",
                    )
                    max_epochs = gr.Number(label="Max epochs", value=200, precision=0, minimum=1)
                    learning_rate = gr.Number(label="Learning rate", value=0.02, precision=6, minimum=1e-6)
                    lambda_train = gr.Number(label="L2 weight", value=0.95, precision=4, minimum=0.0, maximum=1.0)
                    compression = gr.Number(label="Compression", value=0.0001, precision=6, minimum=1e-8)
                    batch_size = gr.Number(label="Batch size", value=160, precision=0, minimum=1)
                    lr_patience = gr.Number(label="LR scheduler patience", value=4, precision=0, minimum=1)

                run_button = gr.Button("Run RFlash", variant="primary")

            with gr.Column(scale=2):
                status = gr.Textbox(label="Status", interactive=False, lines=12, max_lines=30)
                geometry_gallery = gr.Gallery(
                    label="Geometry review",
                    columns=2,
                    height=360,
                    object_fit="contain",
                    visible=False,
                )
                with gr.Row():
                    accept_geometry_button = gr.Button("Accept geometry", visible=False)
                    reject_geometry_button = gr.Button("Reject and try next", visible=False)
                training_curve = gr.Image(label="Training curves", height=360)
                output_gallery = gr.Gallery(label="Representative outputs", columns=3, height=520)
                download_output = gr.File(label="Download processed output")

        input_kind.change(
            _update_selection_controls,
            inputs=[input_kind, probe],
            outputs=[probe, geometry_mode, num_slices, output_format, accepted_types],
        )
        probe.change(
            _update_selection_controls,
            inputs=[input_kind, probe],
            outputs=[probe, geometry_mode, num_slices, output_format, accepted_types],
        )
        advanced.change(
            _toggle_advanced_params,
            inputs=[advanced],
            outputs=[advanced_settings],
        )
        accept_geometry_button.click(
            _accept_geometry_and_maybe_run,
            inputs=[
                uploaded_files,
                server_path,
                input_kind,
                probe,
                output_format,
                output_dir,
                geometry_state,
                advanced,
                max_epochs,
                learning_rate,
                lambda_train,
                compression,
                batch_size,
                lr_patience,
            ],
            outputs=[
                output_gallery,
                download_output,
                status,
                training_curve,
                geometry_state,
                geometry_gallery,
                accept_geometry_button,
                reject_geometry_button,
            ],
        )
        reject_geometry_button.click(
            _reject_geometry_candidate_for_interface,
            inputs=[geometry_state],
            outputs=[
                geometry_state,
                geometry_gallery,
                status,
                accept_geometry_button,
                reject_geometry_button,
            ],
        )
        run_button.click(
            _run_or_prepare_geometry,
            inputs=[
                uploaded_files,
                server_path,
                input_kind,
                probe,
                geometry_mode,
                num_slices,
                output_format,
                output_dir,
                geometry_state,
                advanced,
                max_epochs,
                learning_rate,
                lambda_train,
                compression,
                batch_size,
                lr_patience,
            ],
            outputs=[
                output_gallery,
                download_output,
                status,
                training_curve,
                geometry_state,
                geometry_gallery,
                accept_geometry_button,
                reject_geometry_button,
            ],
        )

    app.queue()
    return app


def _update_selection_controls(input_kind: str, probe: str) -> tuple:
    """Show controls relevant to the chosen input workflow."""

    stack_selected = input_kind == INPUT_STACK
    curvilinear_selected = stack_selected and probe == PROBE_CURVILINEAR
    output_choices = (
        [FORMAT_NII, FORMAT_MHA]
        if input_kind == INPUT_VOLUME
        else [FORMAT_NII, FORMAT_MHA, FORMAT_ZIP_PNG, FORMAT_ZIP_JPG, FORMAT_MP4]
    )
    default_output = FORMAT_NII
    return (
        gr.update(visible=stack_selected),
        gr.update(visible=curvilinear_selected),
        gr.update(visible=curvilinear_selected, value=1),
        gr.update(choices=output_choices, value=default_output),
        _accepted_data_message(input_kind, probe),
    )


def _toggle_advanced_params(advanced: bool) -> gr.Group:
    """Show advanced settings only when requested."""

    return gr.update(visible=advanced)


def _accepted_data_message(input_kind: str, probe: str) -> str:
    """Describe accepted input types for the current selection."""

    if input_kind == INPUT_VOLUME:
        return (
            "Accepted input for `3D ultrasound volume`:\n"
            "- One `.mha`, `.nii`, or `.nii.gz` file.\n"
            "- The processed output can be exported as `.nii.gz` or `.mha`."
        )

    if probe == PROBE_CURVILINEAR:
        return (
            "Accepted input for `2D stack` with a `curvilinear` probe:\n"
            "- One or more image files: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tif`, `.tiff`.\n"
            "- A stack stored as `.mha`, `.nii`, or `.nii.gz`. The last axis is treated as slice index.\n"
            "- Output can be exported as `.nii.gz`, `.mha`, or a zip archive of `.png` or `.jpg` slices."
            "\n- An `.mp4` video can also be exported."
        )

    return (
        "Accepted input for `2D stack` with a `linear` probe:\n"
        "- One or more image files: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tif`, `.tiff`.\n"
        "- A stack stored as `.mha`, `.nii`, or `.nii.gz`.\n"
        "- Synthetic liver-style `.npy` input is also supported.\n"
        "- Output can be exported as `.nii.gz`, `.mha`, a zip archive of `.png` or `.jpg` slices, or an `.mp4` video."
    )


def _format_gui_error(exc: Exception) -> str:
    """Format an exception for the Status panel, including its traceback."""

    message = str(exc) or exc.__class__.__name__
    return f"RFlash failed: {message}\n\n{traceback.format_exc()}"


def _reject_geometry_candidate_for_interface(
    geometry_state: GeometryWorkflowState | None,
) -> tuple[GeometryWorkflowState | None, Any, str, gr.Button, gr.Button]:
    """Run geometry rejection and expose failures in the Status panel."""

    try:
        return _reject_geometry_candidate(geometry_state)
    except Exception as exc:
        return (
            geometry_state,
            gr.update(value=[], visible=False),
            _format_gui_error(exc),
            gr.update(visible=False),
            gr.update(visible=False),
        )


def _prepare_geometry_workflow(
    uploaded_files: list[str] | None,
    server_path: str,
    input_kind: str,
    probe: str,
    geometry_mode: str,
    num_slices: float,
    output_dir: str | None,
) -> tuple[GeometryWorkflowState | None, Any, str, gr.Button, gr.Button]:
    """Estimate scanner geometry and present the current candidate for review."""

    resolved_input = _resolve_input(uploaded_files, server_path)
    output_root = _resolve_output_dir(output_dir)
    overlay_dir = output_root / "intermediate_images"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    if input_kind == INPUT_VOLUME:
        volume, header = load_volume(resolved_input)
        spacing_mm = np.asarray([float(value) for value in header.spacing])
        geometry = estimate_scanner_geometry_volume(volume, spacing_mm, overlay_dir=overlay_dir, verbose=False)
        state = GeometryWorkflowState(
            input_kind=input_kind,
            probe=None,
            geometry_mode=None,
            target_count=1,
            completed=True,
            volume_geometry=geometry,
            overlay_paths=_find_geometry_overlays(overlay_dir),
            volume_data=np.asarray(volume),
            volume_spacing_mm=spacing_mm,
            current_geometry=geometry,
            output_dir=output_root,
        )
        return _geometry_workflow_response(state)

    if probe != PROBE_CURVILINEAR:
        return (
            None,
            gr.update(value=[], visible=False),
            "Linear-probe inputs do not require scanner geometry estimation. You can proceed directly to training.",
            gr.update(visible=False),
            gr.update(visible=False),
        )

    stack = _load_stack_input(resolved_input)
    target_count = _validate_num_slices(num_slices, stack.shape[0])
    state = GeometryWorkflowState(
        input_kind=input_kind,
        probe=probe,
        geometry_mode=geometry_mode,
        target_count=target_count,
        stack_data=stack,
        stack_shape=tuple(stack.shape),
        candidate_indices=list(range(stack.shape[0])),
        output_dir=output_root,
    )
    state = _prepare_next_candidate(state, stack, overlay_dir)
    return _geometry_workflow_response(state)


def _accept_geometry_candidate(
    geometry_state: GeometryWorkflowState | None,
) -> tuple[GeometryWorkflowState | None, Any, str, gr.Button, gr.Button]:
    """Accept the current estimate and advance the review workflow."""

    if geometry_state is None or geometry_state.current_geometry is None:
        raise gr.Error("No geometry candidate is active. Please estimate geometry first.")

    state = geometry_state
    if state.input_kind == INPUT_VOLUME:
        return (
            state,
            gr.update(value=state.overlay_paths, visible=True),
            "Volume geometry accepted. Starting RFlash.",
            gr.update(visible=False),
            gr.update(visible=False),
        )

    if state.geometry_mode == GEOMETRY_SHARED:
        state.accepted_geometries = [state.current_geometry for _ in range(state.target_count)]
        state.accepted_indices = list(range(state.target_count))
        state.accepted_overlays = [state.current_overlay] if state.current_overlay is not None else []
        state.completed = True
        return _geometry_workflow_response(state)

    state.accepted_geometries.append(state.current_geometry)
    state.accepted_indices.append(int(state.current_candidate_index))
    if state.current_overlay is not None:
        state.accepted_overlays.append(state.current_overlay)

    if len(state.accepted_geometries) >= state.target_count:
        state.completed = True
        return _geometry_workflow_response(state)

    stack = _reload_stack_from_state(state)
    overlay_dir = (
        Path(state.current_overlay).parent
        if state.current_overlay is not None
        else Path("outputs/gradio/intermediate_images")
    )
    state = _prepare_next_candidate(state, stack, overlay_dir)
    return _geometry_workflow_response(state)


def _reject_geometry_candidate(
    geometry_state: GeometryWorkflowState | None,
) -> tuple[GeometryWorkflowState | None, Any, str, gr.Button, gr.Button]:
    """Reject the current geometry estimate and move to the next candidate."""

    if geometry_state is None:
        raise gr.Error("No geometry workflow is active. Press Run RFlash to start geometry estimation.")

    if geometry_state.input_kind == INPUT_VOLUME:
        if geometry_state.volume_data is None or geometry_state.volume_spacing_mm is None:
            raise gr.Error("The volume geometry review became stale. Press Run RFlash to restart it.")
        overlay_dir = (
            Path(geometry_state.overlay_paths[0]).parent
            if geometry_state.overlay_paths
            else Path("outputs/gradio/intermediate_images")
        )
        geometry = estimate_scanner_geometry_volume(
            geometry_state.volume_data,
            geometry_state.volume_spacing_mm,
            overlay_dir=overlay_dir,
            verbose=False,
        )
        geometry_state.volume_geometry = geometry
        geometry_state.overlay_paths = _find_geometry_overlays(overlay_dir)
        return _geometry_workflow_response(geometry_state)

    stack = _reload_stack_from_state(geometry_state)
    overlay_dir = (
        Path(geometry_state.current_overlay).parent
        if geometry_state.current_overlay is not None
        else Path("outputs/gradio/intermediate_images")
    )
    state = _prepare_next_candidate(geometry_state, stack, overlay_dir)
    return _geometry_workflow_response(state)


def _prepare_next_candidate(
    state: GeometryWorkflowState,
    stack: np.ndarray,
    overlay_dir: Path,
) -> GeometryWorkflowState:
    """Estimate geometry for the next available slice candidate."""

    while state.current_candidate_position < len(state.candidate_indices):
        slice_idx = state.candidate_indices[state.current_candidate_position]
        state.current_candidate_position += 1

        overlay_path = overlay_dir / f"geometry_slice_{slice_idx:03d}.png"
        try:
            geometry, success = estimate_geometry_slice(
                stack[slice_idx],
                spacing_plane_mm=(1.0, 1.0),
                overlay_path=overlay_path,
                verbose=False,
            )
        except Exception:
            continue
        if success and geometry is not None:
            state.current_candidate_index = int(slice_idx)
            state.current_geometry = geometry
            state.current_overlay = str(overlay_path)
            return state

    raise gr.Error("No more slices were available with a valid geometry estimate.")


def _geometry_workflow_response(
    state: GeometryWorkflowState,
) -> tuple[GeometryWorkflowState, Any, str, gr.Button, gr.Button]:
    """Build the UI response for the current geometry workflow state."""

    if state.input_kind == INPUT_VOLUME:
        return (
            state,
            gr.update(value=state.overlay_paths, visible=True),
            "Estimated 3D scanner geometry. Please inspect it and accept or reject it.",
            gr.update(visible=True),
            gr.update(visible=True),
        )

    if state.completed:
        if state.geometry_mode == GEOMETRY_SHARED:
            status = (
                "Shared geometry accepted. The selected estimate will be used for the requested slices. "
                "You can proceed to training or adjust parameters first."
            )
        else:
            status = (
                f"Accepted {len(state.accepted_geometries)} geometry estimate(s). "
                "The requested curvilinear slices are ready for training."
            )
        return (
            state,
            gr.update(
                value=(
                    [state.accepted_overlays[-1]]
                    if state.accepted_overlays
                    else ([state.current_overlay] if state.current_overlay else [])
                ),
                visible=False,
            ),
            status,
            gr.update(visible=False),
            gr.update(visible=False),
        )

    if state.geometry_mode == GEOMETRY_SHARED:
        status = (
            f"Reviewing candidate slice {state.current_candidate_index}. "
            "Accept to reuse this geometry for the selected stack, or reject to try the next slice."
        )
    else:
        status = (
            f"Accepted {len(state.accepted_geometries)} of {state.target_count} required geometries. "
            f"Reviewing slice {state.current_candidate_index}. Accept to keep it, or reject to try the next slice."
        )

    overlays = [state.current_overlay] if state.current_overlay is not None else []
    return (
        state,
        gr.update(value=overlays, visible=True),
        status,
        gr.update(visible=True),
        gr.update(visible=True),
    )


def _run_or_prepare_geometry(
    uploaded_files: list[str] | None,
    server_path: str | None,
    input_kind: str,
    probe: str,
    geometry_mode: str,
    num_slices: float,
    output_format: str,
    output_dir: str | None,
    geometry_state: GeometryWorkflowState | None,
    advanced: bool,
    max_epochs: float,
    learning_rate: float,
    lambda_train: float,
    compression: float,
    batch_size: float,
    lr_patience: float,
    progress: gr.Progress = gr.Progress(track_tqdm=False),
):
    """Start geometry review when needed, otherwise start shadow removal."""

    try:
        geometry_required = input_kind == INPUT_VOLUME or (input_kind == INPUT_STACK and probe == PROBE_CURVILINEAR)
        if geometry_required and (geometry_state is None or not geometry_state.completed):
            state, gallery, status, accept, reject = _prepare_geometry_workflow(
                uploaded_files,
                server_path,
                input_kind,
                probe,
                geometry_mode,
                num_slices,
                output_dir,
            )
            yield [], None, status, None, state, gallery, accept, reject
            return

        for result in _run_rflash_for_interface(
            uploaded_files,
            server_path,
            input_kind,
            probe,
            output_format,
            output_dir,
            geometry_state,
            advanced,
            max_epochs,
            learning_rate,
            lambda_train,
            compression,
            batch_size,
            lr_patience,
            progress,
        ):
            yield (*result, None, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False))
    except Exception as exc:
        details = _format_gui_error(exc)
        yield [], None, details, None, None, gr.update(visible=False), gr.update(visible=False), gr.update(
            visible=False
        )


def _accept_geometry_and_maybe_run(
    uploaded_files: list[str] | None,
    server_path: str | None,
    input_kind: str,
    probe: str,
    output_format: str,
    output_dir: str | None,
    geometry_state: GeometryWorkflowState | None,
    advanced: bool,
    max_epochs: float,
    learning_rate: float,
    lambda_train: float,
    compression: float,
    batch_size: float,
    lr_patience: float,
    progress: gr.Progress = gr.Progress(track_tqdm=False),
):
    """Accept geometry and immediately run RFlash once review is complete."""

    try:
        state, gallery, status, accept, reject = _accept_geometry_candidate(geometry_state)
        if not state.completed:
            yield [], None, status, None, state, gallery, accept, reject
            return

        for result in _run_rflash_for_interface(
            uploaded_files,
            server_path,
            input_kind,
            probe,
            output_format,
            output_dir,
            state,
            advanced,
            max_epochs,
            learning_rate,
            lambda_train,
            compression,
            batch_size,
            lr_patience,
            progress,
        ):
            yield (*result, None, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False))
    except Exception as exc:
        details = _format_gui_error(exc)
        yield [], None, details, None, geometry_state, gr.update(visible=False), gr.update(visible=False), gr.update(
            visible=False
        )


def _run_rflash_for_interface(
    uploaded_files: list[str] | None,
    server_path: str | None,
    input_kind: str,
    probe: str,
    output_format: str,
    output_dir: str | None,
    geometry_state: GeometryWorkflowState | None,
    advanced: bool,
    max_epochs: float,
    learning_rate: float,
    lambda_train: float,
    compression: float,
    batch_size: float,
    lr_patience: float,
    progress: gr.Progress = gr.Progress(track_tqdm=False),
):
    """Run RFlash and stream progress updates back into the interface."""

    resolved_input = _resolve_input(uploaded_files, server_path)
    output_root = (
        geometry_state.output_dir
        if geometry_state is not None and geometry_state.output_dir is not None
        else _resolve_output_dir(output_dir)
    )
    device = select_device(prefer_cuda=True)
    overrides = (
        None
        if not advanced
        else _collect_training_overrides(
            max_epochs=max_epochs,
            learning_rate=learning_rate,
            lambda_train=lambda_train,
            compression=compression,
            batch_size=batch_size,
            lr_patience=lr_patience,
        )
    )

    if input_kind == INPUT_VOLUME and (geometry_state is None or geometry_state.volume_geometry is None):
        raise gr.Error("Please press Run RFlash to estimate and review the volume geometry first.")
    if (
        input_kind == INPUT_STACK
        and probe == PROBE_CURVILINEAR
        and (geometry_state is None or not geometry_state.completed)
    ):
        raise gr.Error("Please finish the curvilinear geometry review before running RFlash.")

    event_queue: queue.Queue[tuple[str, Any]] = queue.Queue()

    def _callback(event: dict) -> None:
        event_queue.put(("progress", event))

    def _worker() -> None:
        try:
            if input_kind == INPUT_VOLUME:
                result = run_3d_volume_inference(
                    input_path=resolved_input,
                    output_dir=output_root,
                    device=device,
                    silent=True,
                    geometry=geometry_state.volume_geometry,
                    training_overrides=overrides,
                    progress_callback=_callback,
                )
            elif probe == PROBE_CURVILINEAR:
                result = run_curvilinear_stack_inference(
                    input_path=resolved_input,
                    output_dir=output_root,
                    device=device,
                    silent=True,
                    geometry=geometry_state.accepted_geometries,
                    indices=geometry_state.accepted_indices,
                    num_slices=geometry_state.target_count,
                    training_overrides=overrides,
                    progress_callback=_callback,
                )
            else:
                result = run_linear_stack_inference(
                    input_path=resolved_input,
                    output_dir=output_root,
                    device=device,
                    silent=True,
                    training_overrides=overrides,
                    progress_callback=_callback,
                )
            event_queue.put(("done", result))
        except Exception as exc:
            # The worker runs in another thread, so preserve its traceback before
            # handing the failure back to the Gradio request thread.
            event_queue.put(("error", f"{exc}\n\n{traceback.format_exc()}"))

    worker = threading.Thread(target=_worker, daemon=True)
    worker.start()

    status = f"Running RFlash on {device}. Waiting for training to start."
    curve_path = None
    final_curves: tuple[list[float], list[float], list[float]] | None = None
    yield [], None, status, None

    while True:
        try:
            event_type, payload = event_queue.get(timeout=0.5)
        except queue.Empty:
            if not worker.is_alive():
                break
            continue

        if event_type == "error":
            raise RuntimeError(payload)

        if event_type == "progress":
            if payload["phase"] == "batch":
                progress(
                    (payload["epoch"], payload["max_epochs"]),
                    desc=f"Training epoch {payload['epoch']}/{payload['max_epochs']}",
                )
            else:
                final_curves = (payload["losses"], payload["l2s"], payload["ssims"])
            continue

        if event_type == "done":
            result = payload
            if final_curves is not None:
                curve_path = _write_live_training_curve(output_root, final_curves[0], final_curves[1], final_curves[2])
            download_path = _export_processed_output(result, output_format)
            preferred_slice = 88 if _is_fetal_brain_demo_input(resolved_input) else None
            preview_paths = _write_output_previews(result, preferred_volume_slice=preferred_slice)
            status = f"RFlash finished on {device}. Processed output written to {download_path}."
            yield preview_paths, str(download_path), status, curve_path
            return

    raise gr.Error("Training terminated unexpectedly before producing an output.")


def _collect_training_overrides(
    max_epochs: float,
    learning_rate: float,
    lambda_train: float,
    compression: float,
    batch_size: float,
    lr_patience: float,
) -> dict[str, float | int]:
    """Validate manual UI parameter overrides."""

    return {
        "max_epochs": int(max_epochs),
        "lr": float(learning_rate),
        "lamda_train": float(lambda_train),
        "compression": float(compression),
        "batch_size": int(batch_size),
        "lr_shed_patience": int(lr_patience),
    }


def _resolve_input(uploaded_files: list[str] | None, server_path: str | None) -> str | Path | list[str | Path]:
    """Resolve uploaded files or a server-side path to an inference input."""

    server_path = server_path or ""
    if server_path.strip():
        path = Path(server_path.strip()).expanduser()
        if not path.exists():
            raise gr.Error(f"Input path does not exist: {path}")
        return path

    files = _as_path_list(uploaded_files)
    if not files:
        raise gr.Error("Please upload data or enter a server-side path.")
    missing_files = [path for path in files if not path.exists()]
    if missing_files:
        missing_text = ", ".join(str(path) for path in missing_files)
        raise gr.Error(f"Uploaded input path does not exist: {missing_text}")
    if len(files) == 1:
        return files[0]
    return files


def _resolve_output_dir(output_dir: str | None) -> Path:
    """Resolve the requested output directory."""

    if (output_dir or "").strip():
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


def _load_stack_input(input_value: str | Path | list[str | Path]) -> np.ndarray:
    """Load a stack of 2D slices from images, NumPy arrays, or a volume file."""

    if isinstance(input_value, list):
        return load_image_files(input_value)

    path = Path(input_value)
    if path.is_dir():
        if any(path.glob("*.npy")):
            return np.transpose(load_synthetic_liver(path), (2, 0, 1))
        return load_image_directory(path)
    if _is_volume_file(path):
        volume, _ = load_volume(path)
        return np.transpose(np.asarray(volume), (2, 0, 1))
    if path.suffix.lower() == ".npy":
        return np.transpose(load_synthetic_liver(path), (2, 0, 1))
    return load_image_files([path])


def _reload_stack_from_state(state: GeometryWorkflowState) -> np.ndarray:
    """Reload the current stack from the active geometry workflow."""

    if state.stack_shape is None:
        raise gr.Error("No stack information is available for geometry review.")
    if state.stack_data is None:
        raise gr.Error("Geometry review state became stale. Please estimate geometry again.")
    return state.stack_data


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


def _is_volume_file(path: Path) -> bool:
    """Return whether a path is a supported medical image volume."""

    suffix = path.suffix.lower()
    if suffix == ".gz" and len(path.suffixes) >= 2:
        suffix = path.suffixes[-2].lower() + suffix
    return suffix in {".mha", ".nii", ".nii.gz"}


def _is_fetal_brain_demo_input(input_value: str | Path | list[str | Path]) -> bool:
    """Return whether an input is the packaged fetal-brain demonstration file."""

    if isinstance(input_value, list):
        return False
    return Path(input_value).name.lower() == "fetal-brain-demo.mha"


def _write_live_training_curve(output_dir: Path, losses: list[float], l2s: list[float], ssims: list[float]) -> str:
    """Write a live-updating training-curve image and return its path."""

    curve_dir = output_dir / "training_stats_live"
    visualize_stats(losses, l2s, ssims, curve_dir, title="Training Progress")
    return str(curve_dir / "training_stats.png")


def _export_processed_output(result: RFlashOutput, output_format: str) -> Path:
    """Export the processed output in the user-selected format."""

    export_dir = result.output_dir / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)
    processed_u8 = _to_export_uint8(result.shadow_reduced_data)

    if output_format == FORMAT_NII:
        path = export_dir / "shadow_reduced_output.nii.gz"
        save_volume(
            _ensure_volume_for_export(processed_u8),
            path,
            header=result.header if result.is_volume else get_medpy_header(result.spacing_mm),
        )
        return path

    if output_format == FORMAT_MHA:
        path = export_dir / "shadow_reduced_output.mha"
        save_volume(
            _ensure_volume_for_export(processed_u8),
            path,
            header=result.header if result.is_volume else get_medpy_header(result.spacing_mm),
        )
        return path

    if output_format == FORMAT_MP4:
        if processed_u8.ndim == 2:
            slice_arrays = [processed_u8]
        else:
            slice_arrays = [processed_u8[..., idx] for idx in range(processed_u8.shape[-1])]
        return _write_mp4_stack(slice_arrays, export_dir / "shadow_reduced_stack.mp4")

    image_suffix = ".png" if output_format == FORMAT_ZIP_PNG else ".jpg"
    image_dir = export_dir / "shadow_reduced_slices"
    if image_dir.exists():
        shutil.rmtree(image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)

    if processed_u8.ndim == 2:
        slice_arrays = [processed_u8]
    else:
        slice_arrays = [processed_u8[..., idx] for idx in range(processed_u8.shape[-1])]

    for slice_idx, slice_array in enumerate(slice_arrays):
        image_path = image_dir / f"slice_{slice_idx:03d}{image_suffix}"
        Image.fromarray(slice_array).save(image_path, quality=95)

    zip_path = export_dir / (
        "shadow_reduced_slices_png.zip" if image_suffix == ".png" else "shadow_reduced_slices_jpg.zip"
    )
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for image_path in sorted(image_dir.glob(f"*{image_suffix}")):
            archive.write(image_path, arcname=image_path.name)
    return zip_path


def _write_mp4_stack(slice_arrays: list[np.ndarray], output_path: Path, fps: float = 10.0) -> Path:
    """Write a sequence of 8-bit grayscale slices as an MP4 video."""

    if not slice_arrays:
        raise ValueError("Cannot create an MP4 from an empty image stack.")

    first_frame = np.asarray(slice_arrays[0], dtype=np.uint8)
    if first_frame.ndim != 2:
        raise ValueError(f"Expected 2D video frames, got shape {first_frame.shape}.")
    height, width = first_frame.shape
    # Common MP4 codecs require even frame dimensions. Edge padding preserves
    # the image content without changing the displayed intensity range.
    padded_height = height + height % 2
    padded_width = width + width % 2
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (padded_width, padded_height),
        True,
    )
    if not writer.isOpened():
        raise RuntimeError("Could not open an MP4 video writer. Check the available OpenCV codecs.")

    try:
        for slice_array in slice_arrays:
            frame = np.asarray(slice_array, dtype=np.uint8)
            if frame.shape != (height, width):
                raise ValueError("All image stack slices must have the same dimensions for MP4 export.")
            if padded_height != height or padded_width != width:
                frame = np.pad(frame, ((0, padded_height - height), (0, padded_width - width)), mode="edge")
            writer.write(cv2.cvtColor(np.ascontiguousarray(frame), cv2.COLOR_GRAY2BGR))
    finally:
        writer.release()
    return output_path


def _to_export_uint8(image: np.ndarray) -> np.ndarray:
    """Convert a result array to the exported 8-bit representation."""

    image_array = np.asarray(image)
    mask = image_array > 0.01 if image_array.ndim >= 2 else None
    if mask is not None and np.any(mask):
        return to_8bit_graysacle(image_array, volume_mask=mask)
    return to_8bit_graysacle(image_array)


def _ensure_volume_for_export(image: np.ndarray) -> np.ndarray:
    """Ensure volume-style exports always have a slice axis."""

    image_array = np.asarray(image)
    if image_array.ndim == 2:
        return image_array[..., None]
    return image_array


def _write_output_previews(result: RFlashOutput, preferred_volume_slice: int | None = None) -> list[str]:
    """Write representative output previews for the interface."""

    preview_dir = result.output_dir / "web_previews"
    if preview_dir.exists():
        shutil.rmtree(preview_dir)
    preview_dir.mkdir(parents=True, exist_ok=True)

    preview_paths = []
    if result.is_volume:
        for label, volume in (("original", result.original_data), ("shadow_reduced", result.shadow_reduced_data)):
            # Use the same volume-wide conversion as the downloadable processed
            # output. Re-normalizing each preview slice would change its contrast.
            normalize = label != "shadow_reduced"
            preview_volume = _to_export_uint8(volume) if not normalize else volume
            for axis, slice_idx, image, aspect in _representative_volume_slices(
                preview_volume, result.spacing_mm, preferred_slice_index=preferred_volume_slice
            ):
                output_path = preview_dir / f"{label}_{axis}_{slice_idx:03d}.png"
                _save_preview_image(
                    image,
                    output_path,
                    f"{label.replace('_', ' ')} {axis} slice {slice_idx}",
                    aspect=aspect,
                    normalize=normalize,
                    rotate_90=True,
                )
                preview_paths.append(str(output_path))
        return preview_paths

    selected_slices = _representative_stack_slices(result.original_data, result.spacing_mm)
    for label, stack in (("original", result.original_data), ("shadow_reduced", result.shadow_reduced_data)):
        normalize = label != "shadow_reduced"
        preview_stack = _to_export_uint8(stack) if not normalize else stack
        for slice_idx, _, aspect in selected_slices:
            image = preview_stack if preview_stack.ndim == 2 else preview_stack[..., slice_idx]
            output_path = preview_dir / f"{label}_slice_{slice_idx:03d}.png"
            _save_preview_image(
                image,
                output_path,
                f"{label.replace('_', ' ')} slice {slice_idx}",
                aspect=aspect,
                normalize=normalize,
                rotate_90=False,
            )
            preview_paths.append(str(output_path))
    return preview_paths


def _representative_volume_slices(
    volume: np.ndarray,
    spacing_mm: np.ndarray | list[float],
    preferred_slice_index: int | None = None,
) -> list[tuple[str, int, np.ndarray, float]]:
    """Return one centre slice for each of the three volume dimensions."""

    volume_array = np.asarray(volume)
    spacing = np.asarray(spacing_mm, dtype=float)
    results = []
    axis_names = ("axis0", "axis1", "axis2")
    reverse = False
    for axis, axis_name in enumerate(axis_names):
        mid = volume_array.shape[axis] // 2
        if axis == 1 and preferred_slice_index is not None:
            mid = min(max(int(preferred_slice_index), 0), volume_array.shape[axis] - 1)
            reverse = True
        if axis == 0:
            image = volume_array[mid, :, :]
            aspect = _spacing_aspect(spacing[1], spacing[2])
        elif axis == 1:
            if reverse:
                image = np.flipud(volume_array[:, mid, :])
            else:
                image = volume_array[:, mid, :]
            aspect = _spacing_aspect(spacing[0], spacing[2])
        else:
            image = volume_array[:, :, mid]
            aspect = _spacing_aspect(spacing[0], spacing[1])
        results.append((axis_name, mid, image, aspect))
    return results


def _representative_stack_slices(
    stack: np.ndarray,
    spacing_mm: np.ndarray | list[float],
) -> list[tuple[int, np.ndarray, float]]:
    """Return three reproducibly selected representative stack slices."""

    stack_array = np.asarray(stack)
    spacing = np.asarray(spacing_mm, dtype=float)
    aspect = _spacing_aspect(spacing[0], spacing[1]) if spacing.size >= 2 else 1.0
    if stack_array.ndim == 2:
        return [(0, stack_array, aspect)]
    last_axis = stack_array.shape[-1]
    rng = np.random.default_rng()
    indices = sorted(rng.choice(last_axis, size=min(3, last_axis), replace=False).tolist())
    return [(idx, stack_array[..., idx], aspect) for idx in indices]


def _save_preview_image(
    image: np.ndarray,
    output_path: Path,
    title: str,
    aspect: float = 1.0,
    normalize: bool = True,
    rotate_90: bool = False,
) -> None:
    """Save one grayscale preview image."""

    fig, ax = plt.subplots(figsize=(6, 5))
    if rotate_90:
        image = np.rot90(image, k=-1)
        aspect = 1.0 / aspect if aspect else 1.0
    display_image = normalize_for_display(image) if normalize else image
    ax.imshow(display_image, cmap="gray", aspect=aspect)
    ax.set_title(title)
    ax.axis("off")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _spacing_aspect(row_spacing: float, col_spacing: float) -> float:
    """Return a display aspect ratio from physical spacing values."""

    if row_spacing <= 0 or col_spacing <= 0:
        return 1.0
    return float(row_spacing / col_spacing)


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
