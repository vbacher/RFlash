###### imports ######

## library imports
import numpy as np
import torch


## project imports
from src.datasets import Dataset_3D_volume, Dataset_2D_lin_array
from src.model.representation import ExplicitRepresentation, SlicePoses
from src.model.rendering import Render_engine


###### body ######


def _accumulate_inverse_trilinear(
    coord_array: torch.Tensor,
    values: torch.Tensor,
    volume_shape: tuple,
    accumulator_values: torch.Tensor,
    accumulator_weights: torch.Tensor,
) -> None:
    """Accumulate weighted trilinear pseudo-inverse contributions into global buffers."""

    # FIXME: not yet tested

    coords = (coord_array + 3).flatten(0, -2)
    values = values.flatten()

    mask_x = torch.logical_and(coords[:, 0] <= volume_shape[0] + 2, coords[:, 0] >= 0)
    mask_y = torch.logical_and(coords[:, 1] <= volume_shape[1] + 2, coords[:, 1] >= 0)
    mask_z = torch.logical_and(coords[:, 2] <= volume_shape[2] + 2, coords[:, 2] >= 0)
    mask = torch.logical_and(mask_x, torch.logical_and(mask_y, mask_z))

    if not mask.any():
        return

    coords = coords[mask, :]
    values = values[mask]

    x_floor = torch.floor(coords[:, 0]).int()
    y_floor = torch.floor(coords[:, 1]).int()
    z_floor = torch.floor(coords[:, 2]).int()
    x_ceil = x_floor + 1
    y_ceil = y_floor + 1
    z_ceil = z_floor + 1

    dep = torch.cat(
        [
            torch.vstack([x_floor, y_floor, z_floor])[..., None],
            torch.vstack([x_floor, y_ceil, z_ceil])[..., None],
            torch.vstack([x_floor, y_floor, z_ceil])[..., None],
            torch.vstack([x_floor, y_ceil, z_floor])[..., None],
            torch.vstack([x_ceil, y_floor, z_floor])[..., None],
            torch.vstack([x_ceil, y_ceil, z_ceil])[..., None],
            torch.vstack([x_ceil, y_floor, z_ceil])[..., None],
            torch.vstack([x_ceil, y_ceil, z_floor])[..., None],
        ],
        dim=2,
    ).reshape((3, values.shape[0] * 8))

    weights = torch.vstack(
        [
            (x_ceil - coords[:, 0]) * (y_ceil - coords[:, 1]) * (z_ceil - coords[:, 2]),
            (coords[:, 0] - x_ceil) * (coords[:, 1] - y_floor) * (z_floor - coords[:, 2]),
            (coords[:, 0] - x_ceil) * (y_ceil - coords[:, 1]) * (z_floor - coords[:, 2]),
            (x_ceil - coords[:, 0]) * (coords[:, 1] - y_floor) * (z_ceil - coords[:, 2]),
            (x_floor - coords[:, 0]) * (y_ceil - coords[:, 1]) * (coords[:, 2] - z_ceil),
            (coords[:, 0] - x_floor) * (coords[:, 1] - y_floor) * (coords[:, 2] - z_floor),
            (coords[:, 0] - x_floor) * (y_ceil - coords[:, 1]) * (coords[:, 2] - z_floor),
            (x_floor - coords[:, 0]) * (coords[:, 1] - y_floor) * (coords[:, 2] - z_ceil),
        ]
    ).reshape((values.shape[0] * 8))

    values = torch.cat([values[:, None] for _ in range(8)], dim=1).flatten()

    if coord_array.device.type == "mps":
        unique_points, idxs = torch.unique(dep.to(device="cpu"), dim=1, return_inverse=True)
        unique_points = unique_points.to(device=coord_array.device)
        idxs = idxs.to(device=coord_array.device)
    else:
        unique_points, idxs = torch.unique(dep, dim=1, return_inverse=True)

    unique_values = torch.zeros(unique_points.shape[1], device=coord_array.device)
    unique_weights = torch.zeros(unique_points.shape[1], device=coord_array.device)
    unique_values.index_add_(0, idxs, values * weights)
    unique_weights.index_add_(0, idxs, weights)

    accumulator_values.index_put_(tuple(unique_points), unique_values, accumulate=True)
    accumulator_weights.index_put_(tuple(unique_points), unique_weights, accumulate=True)


def _accumulate_inverse_bilinear(
    coord_array: torch.Tensor,
    values: torch.Tensor,
    imgstack_shape: tuple,
    accumulator_values: torch.Tensor,
    accumulator_weights: torch.Tensor,
) -> None:
    """Accumulate weighted bilinear pseudo-inverse contributions into global buffers."""

    if values.shape != coord_array.shape[:-1]:
        raise ValueError(
            "Mismatch between rendered values and coordinates. "
            f"values shape={tuple(values.shape)}, coord shape={tuple(coord_array.shape)}"
        )

    coords = coord_array.flatten(0, -2).clone()
    vals = values.flatten()

    # Match stack interpolation convention: pad x/y by 3 voxels, keep z unpadded.
    coords[:, :2] += 3

    x = coords[:, 0]
    y = coords[:, 1]
    z = coords[:, 2].round().long()

    x_floor = torch.floor(x).long()
    y_floor = torch.floor(y).long()
    x_ceil = x_floor + 1
    y_ceil = y_floor + 1

    mask_x = torch.logical_and(x >= 0, x <= imgstack_shape[0] + 2)
    mask_y = torch.logical_and(y >= 0, y <= imgstack_shape[1] + 2)
    mask_z = torch.logical_and(z >= 0, z < imgstack_shape[2])
    mask = torch.logical_and(mask_z, torch.logical_and(mask_x, mask_y))

    if not mask.any():
        return

    x = x[mask]
    y = y[mask]
    z = z[mask]
    vals = vals[mask]
    x_floor = x_floor[mask]
    y_floor = y_floor[mask]
    x_ceil = x_ceil[mask]
    y_ceil = y_ceil[mask]

    x_idx = torch.cat([x_floor, x_floor, x_ceil, x_ceil], dim=0)
    y_idx = torch.cat([y_floor, y_ceil, y_floor, y_ceil], dim=0)
    z_idx = torch.cat([z, z, z, z], dim=0)

    weights = torch.cat(
        [
            (x_ceil - x) * (y_ceil - y),
            (x_ceil - x) * (y - y_floor),
            (x - x_floor) * (y_ceil - y),
            (x - x_floor) * (y - y_floor),
        ],
        dim=0,
    )
    vals_weighted = torch.cat([vals, vals, vals, vals], dim=0) * weights

    accumulator_values.index_put_((x_idx, y_idx, z_idx), vals_weighted, accumulate=True)
    accumulator_weights.index_put_((x_idx, y_idx, z_idx), weights, accumulate=True)


def render_volume(
    representation_model: ExplicitRepresentation,
    pose_model: SlicePoses,
    render_model: Render_engine,
    dataset: Dataset_3D_volume | Dataset_2D_lin_array,
    batch_size: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Renders a volume

    Args:
        representation_model (ExplicitRepresentation): Trained representation model
        pose_model (LearnPose): Pose model
        render_model (Render_engine_3D): Render engine
        dataset (Dataset_3D_volume): Dataset
        shadow_reduction (bool, optional): Swith for shadow reduction. Defaults to True.

    Returns:
        torch.Tensor: Rendered volume.
    """

    # shortcut for shadow reduction and two parameters
    # if shadow_reduction and representation_model.dim_latent_space == 2:
    #     return representation_model.get_cost_vol()[...,1]

    device = representation_model.get_device()

    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}")

    # get index of example slice
    idxs = torch.arange(pose_model.num_cams)

    if isinstance(dataset, Dataset_3D_volume):
        padded_shape = tuple(np.asarray(dataset.vh.shape) + 6)
        crop = (slice(3, -3), slice(3, -3), slice(3, -3))
        accumulate_fn = _accumulate_inverse_trilinear
        output_shape = dataset.vh.shape
    elif isinstance(dataset, Dataset_2D_lin_array):
        padded_shape = tuple(np.asarray(dataset.stackh.shape) + np.asarray([6, 6, 0]))
        crop = (slice(3, -3), slice(3, -3), slice(None))
        accumulate_fn = _accumulate_inverse_bilinear
        output_shape = dataset.stackh.shape
    else:
        raise NotImplementedError(f"Unsupported dataset type: {type(dataset)}")

    render_numerator = torch.zeros(padded_shape, device=device)
    render_denominator = torch.zeros(padded_shape, device=device)

    shadow_numerator = torch.zeros(padded_shape, device=device)
    shadow_denominator = torch.zeros(padded_shape, device=device)

    with torch.no_grad():
        for start_idx in range(0, len(idxs), batch_size):
            end_idx = start_idx + batch_size
            idx_batch = idxs[start_idx:end_idx]

            # retrieve coordinates for current batch
            aff_mat_slice = pose_model(idx_batch).to(device=device)
            slice_coords = dataset.sh.get_cart_coord_FoV(aff_mat_slice).to(device=device)

            # render current batch
            parameter_maps = representation_model(slice_coords).detach()
            render_batch, shadow_batch = render_model.rend_wo_shadow(parameter_maps)

            accumulate_fn(
                slice_coords,
                render_batch,
                output_shape,
                render_numerator,
                render_denominator,
            )
            if shadow_batch is not None:
                accumulate_fn(
                    slice_coords,
                    shadow_batch,
                    output_shape,
                    shadow_numerator,
                    shadow_denominator,
                )

    render = render_numerator[crop] / torch.clamp(render_denominator[crop], min=1e-9)
    render[render_denominator[crop] <= 0] = 0

    shadow_mask = shadow_numerator[crop] / torch.clamp(shadow_denominator[crop], min=1e-9)
    shadow_mask[shadow_denominator[crop] <= 0] = 0

    return render.cpu().numpy(), shadow_mask.cpu().numpy()
