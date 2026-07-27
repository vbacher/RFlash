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
    Loss functions used to train the RFlash explicit representation.
License:
    This file is part of the RFlash project and is distributed under the
    repository's LICENSE. See the LICENSE file in the repository root for
    licensing information.
------------------------------------------------------------------------------"""

import torch
from torch.nn import MSELoss
from torchmetrics.image import StructuralSimilarityIndexMeasure


class L2SSIMLoss(torch.nn.Module):
    """Weighted objective combining pixel-wise L2 error and SSIM similarity."""

    def __init__(self, l: float = 0.1, device: torch.device = torch.device("cpu")):
        """Initialize the combined L2/SSIM loss.

        Args:
            l: Weight assigned to the L2 term. The SSIM loss receives
                ``1 - l``.
            device: Execution device for the SSIM metric state.
        """

        super().__init__()

        self.mse = MSELoss()
        self.metric_ssim = StructuralSimilarityIndexMeasure().to(device=device)
        self.l = l

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute weighted loss, L2 component, and SSIM metric.

        Args:
            output: Rendered slices with shape ``(batch, height, width)``.
            target: Observed slices with shape ``(batch, height, width)``.

        Returns:
            Tuple ``(loss, l2, ssim)``. ``loss`` is minimized; ``ssim`` is
            returned for logging.
        """
        l2norm = self.mse(output, target)
        # TorchMetrics expects image tensors as (N, C, H, W). The renderer
        # produces single-channel slices, so a channel axis is inserted here.
        output_aug = output[:, :, :, None]
        output_aug = torch.swapaxes(output_aug, 1, 3)
        target_aug = target[:, :, :, None]
        target_aug = torch.swapaxes(target_aug, 1, 3)
        ssim = self.metric_ssim(output_aug, target_aug)
        ssim_loss = 1 - ssim
        loss = self.l * l2norm + (1 - self.l) * ssim_loss
        return loss, l2norm, ssim
