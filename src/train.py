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
    Training loop for fitting the explicit RFlash representation to observed
    ultrasound slices using the differentiable renderer.
License:
    This file is part of the RFlash project and is distributed under the
    repository's LICENSE. See the LICENSE file in the repository root for
    licensing information.
------------------------------------------------------------------------------"""

import numpy as np
import torch
from tqdm import tqdm

from src.datasets import Dataset_2D_lin_array, Dataset_3D_volume
from src.model.losses import L2SSIMLoss
from src.model.rendering import Render_engine
from src.model.representation import ExplicitRepresentation, SlicePoses
from src.trainings_params import (
    Parameter_Demo2D_curvylinear,
    Parameter_Demo2D_linear,
    Parameter_Demo3D,
)

def train_model(
    representation_model: ExplicitRepresentation,
    pose_model: SlicePoses,
    render_model: Render_engine,
    data: Dataset_3D_volume | Dataset_2D_lin_array,
    training_params: Parameter_Demo3D | Parameter_Demo2D_curvylinear | Parameter_Demo2D_linear,
    device: torch.device,
) -> tuple[list[float], list[float], list[float]]:
    """Train the explicit representation against observed ultrasound slices.

    Args:
        representation_model: Trainable attenuation/scatter representation.
        pose_model: Fixed slice-pose model returning affine matrices.
        render_model: Differentiable ultrasound renderer.
        data: Dataset returning ``(index, label, mask)`` batches.
        training_params: Demo hyperparameters and data-loader settings.
        device: PyTorch device used for training.

    Returns:
        Three lists containing epoch-wise total loss, L2 loss, and SSIM values.
    """

    # Transfer all modules once before creating tensors inside the loop.
    representation_model = representation_model.to(device=device)
    pose_model = pose_model.to(device=device)
    render_model = render_model.to(device=device)

    training_generator = torch.utils.data.DataLoader(data, **training_params.params_generator)

    crit = L2SSIMLoss(l=training_params.lamda_train, device=device)

    optimizer = torch.optim.Adam(
        representation_model.parameters(),
        lr=training_params.lr,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=training_params.lr_shed_patience,
        threshold=1e-8,
    )

    # start training
    losses = []
    l2s = []
    ssims = []
    print("\nStart training model.\n")
    for epoch in tqdm(range(training_params.max_epochs), desc="Train Model"):

        # Keep simple Python lists instead of external experiment tracking so
        # the public demo remains lightweight and reproducible.
        loss_epoch_train = []
        l2_epoch_train = []
        ssim_epoch_train = []

        representation_model.train()
        pose_model.train()

        for idx, label, mask in training_generator:
            label = label.to(device=device)
            mask = mask.to(device=device)

            # Retrieve fixed slice poses for this batch and resample the
            # trainable representation at the corresponding coordinates.
            aff_trans_mat = pose_model(idx).to(device=device)
            slice_coords = training_generator.dataset.sh.get_cart_coord_FoV(aff_trans_mat)

            param_maps = representation_model(slice_coords)

            render = render_model(param_maps)

            # Background pixels do not belong to the ultrasound fan support and
            # would otherwise dominate the objective.
            render[~mask] = 0
            loss, l2, ssim = crit(render, label)

            loss_epoch_train.append(loss.detach().item())
            l2_epoch_train.append(l2.detach().item())
            ssim_epoch_train.append(ssim.detach().item())

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            del label, mask, aff_trans_mat, slice_coords, param_maps, render, loss, l2, ssim

        epoch_loss = float(np.mean(loss_epoch_train))
        epoch_l2 = float(np.mean(l2_epoch_train))
        epoch_ssim = float(np.mean(ssim_epoch_train))
        scheduler.step(epoch_loss)
        losses.append(epoch_loss)
        l2s.append(epoch_l2)
        ssims.append(epoch_ssim)

    return losses, l2s, ssims
