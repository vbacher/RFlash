###### imports ######

import torch
from torch.nn import MSELoss
from torchmetrics.image import StructuralSimilarityIndexMeasure

# FIXME: remove in final version. ONly for debugging

###### body ######


class L2SSIMLoss(torch.nn.Module):
    def __init__(self, l: float = 0.1, device: torch.device = torch.device("cpu")):
        """This loss function is a weighted sum of SSIM and L2 loss

        Args:
            l (float, optional): weighting factor. Defaults to 0.1.
            device (_type_, optional): Execution device. Defaults to torch.device('cpu').
        """

        super().__init__()

        self.mse = MSELoss()
        self.metric_ssim = StructuralSimilarityIndexMeasure().to(device=device)
        self.l = l

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Overload of forward function calculating the weighted L2 and SSIM loss

        Args:
            output (torch.Tensor): Prediction
            target (torch.Tensor): Target

        Returns:
            tuple[torch.Tensor,torch.Tensor,torch.Tensor]: loss, l2 , ssim
        """
        # l2norm = torch.norm(target-output,2)
        l2norm = self.mse(output, target)
        output_aug = output[:, :, :, None]
        output_aug = torch.swapaxes(output_aug, 1, 3)
        target_aug = target[:, :, :, None]
        target_aug = torch.swapaxes(target_aug, 1, 3)
        ssim = self.metric_ssim(output_aug, target_aug)
        ssim_loss = 1 - ssim
        loss = self.l * l2norm + (1 - self.l) * ssim_loss
        return loss, l2norm, ssim
