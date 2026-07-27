###### imports ######

## library imports
import numpy as np
import torch
from tqdm import tqdm

## project imports
from src.datasets import Dataset_2D_lin_array, Dataset_3D_volume
from src.model.losses import L2SSIMLoss
from src.model.rendering import Render_engine
from src.model.representation import ExplicitRepresentation, SlicePoses
from src.trainings_params import (
    Parameter_Demo2D_curvylinear,
    Parameter_Demo2D_linear,
    Parameter_Demo3D,
)

###### body ######


def train_model(
    representation_model: ExplicitRepresentation,
    pose_model: SlicePoses,
    render_model: Render_engine,
    data: Dataset_3D_volume | Dataset_2D_lin_array,
    training_params: Parameter_Demo3D | Parameter_Demo2D_curvylinear | Parameter_Demo2D_linear,
    device: torch.device,
) -> tuple[list[float], list[float], list[float]]:
    """Training function for RFlash model

    Args:
        representation_model (ExplicitRepresentation): Representation model
        pose_model (LearnPose): Pose model
        render_model (Render_engine_3D): Render engine
        data (Dataset_3D_volume): Dataset
        training_params (dict): Dictionary containing training parameter
        verbose (bool, optional): Switch for verbose output. Defaults to False.
        logging (bool, optional): Switch for logging with mlflow. Defaults to True.
    """

    # transfer to device
    representation_model = representation_model.to(device=device)
    pose_model = pose_model.to(device=device)
    render_model = render_model.to(device=device)

    # data generator
    training_generator = torch.utils.data.DataLoader(data, **training_params.params_generator)

    # define loss function
    crit = L2SSIMLoss(l=training_params.lamda_train, device=device)

    # optimizer
    optimizer = torch.optim.Adam(
        representation_model.parameters(),
        lr=training_params.lr,
    )

    sheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
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

        # metrics for tracking
        loss_epoch_train = []
        l2_epoch_train = []
        ssim_epoch_train = []

        # set training parameter correctly
        representation_model.train()
        pose_model.train()

        # Iterate over batches
        for idx, label, mask in training_generator:
            label = label.to(device=device)
            mask = mask.to(device=device)

            # get orientations for sloices
            aff_trans_mat = pose_model(idx).to(device=device)
            slice_coords = training_generator.dataset.sh.get_cart_coord_FoV(aff_trans_mat)

            # retrieve parameters
            param_maps = representation_model(slice_coords)

            # render
            render = render_model(param_maps)

            # calculate loss
            render[~mask] = 0
            loss, l2, ssim = crit(render, label)

            loss_epoch_train.append(loss.detach().item())
            l2_epoch_train.append(l2.detach().item())
            ssim_epoch_train.append(ssim.detach().item())

            # update parameter
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            del label, mask, aff_trans_mat, slice_coords, param_maps, render, loss, l2, ssim

        # update scheduler
        epoch_loss = float(np.mean(loss_epoch_train))
        epoch_l2 = float(np.mean(l2_epoch_train))
        epoch_ssim = float(np.mean(ssim_epoch_train))
        sheduler.step(epoch_loss)
        losses.append(epoch_loss)
        l2s.append(epoch_l2)
        ssims.append(epoch_ssim)

    return losses, l2s, ssims
