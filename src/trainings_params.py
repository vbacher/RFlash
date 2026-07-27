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
    Small parameter objects used by the public demos to configure training,
    rendering, initialization, and data loading.
License:
    This file is part of the RFlash project and is distributed under the
    repository's LICENSE. See the LICENSE file in the repository root for
    licensing information.
------------------------------------------------------------------------------"""

import numpy as np


class TrainingParams:
    """Base training and rendering configuration shared by all demos.

    Args:
        exp_name: Human-readable experiment name used for identifying a demo
            configuration.

    Attributes:
        init_values: Constant initialization values used for the explicit
            representation and renderer.
        lamda_train: Weight of the L2 term in the combined L2/SSIM objective.
        max_epochs: Number of training epochs.
        lr_shed_patience: Patience used by ``ReduceLROnPlateau``.
        lr: Adam learning rate.
        compression: Log-compression parameter used by the renderer.
        params_generator: Keyword arguments passed to ``torch.utils.data.DataLoader``.
    """

    def __init__(self, exp_name: str) -> None:
        """Initialize shared demo hyperparameters.

        Args:
            exp_name: Human-readable experiment name.
        """

        self.exp_name: str = exp_name
        self.init_values: list = [1.8, 4.25, 0.05, 1, 1, 1]
        self.lamda_train: float = 0.95
        self.max_epochs: int = 200
        self.lr_shed_patience: int = 4
        self.lr: float = 0.05
        self.compression: float = 0.0001
        self.params_generator = {
            "batch_size": 160,
            "shuffle": True,
            "num_workers": 0,
            "drop_last": False,
            "pin_memory": False,
        }


class Parameter_Demo3D(TrainingParams):
    """Training configuration for the fetal-brain 3D volume demo."""

    def __init__(self) -> None:
        """Initialize fetal-brain demo defaults."""

        super().__init__("fetal brain")

        # can be set manually but default values are chosen based on volume dimensions
        self.image_size_polar: np.ndarray | None = None
        self.image_size_cartesian: np.ndarray | None = None
        self.num_fan_slices: int | None = None


class Parameter_Demo2D_curvylinear(TrainingParams):
    """Training configuration for curvilinear 2D abdominal ultrasound stacks."""

    def __init__(self) -> None:
        """Initialize curvilinear 2D demo defaults."""

        super().__init__("synthetic liver")

        # can be set manually but default values are chosen based on volume dimensions
        self.image_size_polar: np.ndarray | None = None
        self.image_size_cartesian: np.ndarray | None = None
        self.lr = 0.02


class Parameter_Demo2D_linear(TrainingParams):
    """Training configuration for linear-probe synthetic liver image stacks."""

    def __init__(self) -> None:
        """Initialize linear 2D demo defaults."""

        super().__init__("synthetic liver")

        self.lr = 0.02
