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
    Initialization routines for the explicit attenuation and scatter parameter
    maps used by the RFlash representation.
License:
    This file is part of the RFlash project and is distributed under the
    repository's LICENSE. See the LICENSE file in the repository root for
    licensing information.
------------------------------------------------------------------------------"""

import torch


def Normal_initialization(shape_cost_vol: tuple, constant_init_vals: list[float]) -> torch.Tensor:
    """Return attenuation and scatter parameter maps with small Gaussian noise.

    Args:
        shape_cost_vol: Spatial shape of the unpadded cost volume as
            ``(height, width, depth)``.
        constant_init_vals: Constant initialization values. The first two values
            initialize attenuation and scatter respectively.

    Returns:
        Parameter maps with shape ``(*shape_cost_vol, 2)``.
    """
    init_list = []

    # A tiny standard deviation keeps the public demo close to the intended
    # constant initialization while avoiding a perfectly flat optimization
    # surface at the first step.
    init_list.append(
        torch.empty(shape_cost_vol).normal_(constant_init_vals[0], constant_init_vals[0] * 0.0005)[:, :, :, None]
    )  # 0.0005

    # init scatter
    init_list.append(
        torch.empty(shape_cost_vol).normal_(constant_init_vals[1], constant_init_vals[1] * 0.0005)[:, :, :, None]
    )  # 0.0005

    # combine
    init_vol = torch.cat(tuple(init_list), dim=3)

    return init_vol
