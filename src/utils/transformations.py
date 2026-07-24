import numpy as np


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
