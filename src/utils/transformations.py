import numpy as np
import torch


def standardize_volume(volume: np.ndarray, mean: float, mask: np.ndarray | None = None) -> np.ndarray:

    if mask is not None:
        volume_mask = mask
    else:
        volume_mask = volume > 0

    volume = volume - np.mean(volume[volume_mask])
    volume = volume / np.std(volume[volume_mask])
    volume = volume * (mean / 3)
    volume = volume + mean
    volume[volume < 0] = 0
    volume = volume.astype(np.float32)
    return volume


def rotmat_from_euler(thetas: torch.Tensor) -> torch.Tensor:
    """Gets 4X4 homogenious rotation matrices from euler angles.

    Args:
        thetas (torch.Tensor): Tensor containing n sets of eulerangels with rotations around x,y,z. Dimensions: (n,3)

    Returns:
        torch.tensor: retuns set of 4X4 rotations matrices. Dimensions: (n,3,3).
    """

    assert len(thetas.shape) == 2, "Most likly batch dimension missing."
    assert thetas.shape[1] == 3, "Can only handel 3D rotations but got {thetas.shape[1]}D."
    bs = thetas.shape[0]
    zero = torch.zeros((bs, 1), dtype=torch.float32, device=thetas.device)
    one = torch.ones((bs, 1), dtype=torch.float32, device=thetas.device)

    # rotations around x
    x = torch.cat(
        [
            one,
            zero,
            zero,
            zero,
            zero,
            torch.cos(thetas[:, 0:1]),
            -torch.sin(thetas[:, 0:1]),
            zero,
            zero,
            torch.sin(thetas[:, 0:1]),
            torch.cos(thetas[:, 0:1]),
            zero,
            zero,
            zero,
            zero,
            one,
        ],
        dim=1,
    ).reshape((bs, 4, 4))

    # rotations around y
    y = torch.cat(
        [
            torch.cos(thetas[:, 1:2]),
            zero,
            torch.sin(thetas[:, 1:2]),
            zero,
            zero,
            one,
            zero,
            zero,
            -torch.sin(thetas[:, 1:2]),
            zero,
            torch.cos(thetas[:, 1:2]),
            zero,
            zero,
            zero,
            zero,
            one,
        ],
        dim=1,
    ).reshape((bs, 4, 4))

    # rotations around z
    z = torch.cat(
        [
            torch.cos(thetas[:, 2:3]),
            -torch.sin(thetas[:, 2:3]),
            zero,
            zero,
            torch.sin(thetas[:, 2:3]),
            torch.cos(thetas[:, 2:3]),
            zero,
            zero,
            zero,
            zero,
            one,
            zero,
            zero,
            zero,
            zero,
            one,
        ],
        dim=1,
    ).reshape((bs, 4, 4))

    # combine
    Rs = torch.bmm(z, torch.bmm(y, x))

    return Rs


def transform_slice(img_slice: torch.Tensor, aff_trans_mats: torch.Tensor) -> torch.Tensor:
    """transforms coordinates of slice by applying affine transformation matrices

    Args:
        img_slice (torch.Tensor): _description_
        aff_trans_mats (torch.Tensor): _description_

    Returns:
        torch.Tensor: _description_
    """
    slice_shape = img_slice.shape

    assert img_slice.shape == torch.Size(
        (slice_shape[0], slice_shape[1], slice_shape[2], 3)
    ), "image slice needs to be of shape (batch size, fan radial range, fan angular range, 3)"
    assert aff_trans_mats.shape == torch.Size(
        (slice_shape[0], 4, 4)
    ), "The matrix arraz needs to be of shape (batch_size, 4, 4)."

    # get homogenious coordinates
    transformed = torch.ones((*slice_shape[:3], 4)).to(device=img_slice.device)
    transformed[:, :, :, :3].copy_(img_slice)

    # transform
    transformed = torch.einsum("bmnj, bij -> bmni", transformed, aff_trans_mats)

    return transformed[:, :, :, :3]
