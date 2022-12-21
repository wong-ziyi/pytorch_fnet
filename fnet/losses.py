"""Loss functions for fnet models."""


from typing import Optional

import torch
from torch.nn import functional as F


class HeteroscedasticLoss(torch.nn.Module):
    """Loss function to capture heteroscedastic aleatoric uncertainty."""

    def forward(self, y_hat_batch: torch.Tensor, y_batch: torch.Tensor):
        """Calculate loss.

        Parameters
        ----------
        y_hat_batch
           Batched, 2-channel model output.
        y_batch
           Batched, 1-channel target output.

        """
        mean_batch = y_hat_batch[:, 0:1, :, :, :]
        log_var_batch = y_hat_batch[:, 1:2, :, :, :]
        loss_batch = (
            0.5 * torch.exp(-log_var_batch) * (mean_batch - y_batch).pow(2)
            + 0.5 * log_var_batch
        )
        return loss_batch.mean()


class WeightedMSE(torch.nn.Module):
    """Criterion for weighted mean-squared error."""

    def forward(
        self,
        y_hat_batch: torch.Tensor,
        y_batch: torch.Tensor,
        weight_map_batch: Optional[torch.Tensor] = None,
    ):
        """Calculate weighted MSE.

        Parameters
        ----------
        y_hat_batch
            Batched prediction.
        y_batch
            Batched target.
        weight_map_batch
            Optional weight map.

        """
        if weight_map_batch is None:
            return F.mse_loss(y_hat_batch, y_batch)
        dim = tuple(range(1, len(weight_map_batch.size())))
        return (weight_map_batch * (y_hat_batch - y_batch) ** 2).sum(dim=dim).mean()
    
    
class HuberLoss(torch.nn.Module):
    """Huber loss."""

    def forward(
        self,
        y_hat_batch: torch.Tensor,
        y_batch: torch.Tensor,
        weight_map_batch: Optional[torch.Tensor] = None,
        delta: float = 1.0
    ):
        """Calculate Huber loss.

        Parameters
        ----------
        y_hat_batch
            Batched prediction.
        y_batch
            Batched target.
        weight_map_batch
            Optional weight map.
        delta
            Threshold at which to change between delta-scaled L1 and L2 loss.
        """
        if weight_map_batch is None:
            return (delta ** 2 * (torch.sqrt(1 + ((y_hat_batch - y_batch) / delta) ** 2) - 1)).mean()
        dim = tuple(range(1, len(weight_map_batch.size())))
        # weight map should sum up to 1.0 across all pixels in each image
        return (weight_map_batch * (delta ** 2 * (torch.sqrt(1 + ((y_hat_batch - y_batch) / delta) ** 2) - 1))).sum(dim=dim).mean()
    
    
class SpectralLoss(torch.nn.Module):
    """Spectral loss."""

    def forward(
        self,
        y_hat_batch: torch.Tensor,
        y_batch: torch.Tensor,
        weight_map_batch: Optional[torch.Tensor] = None
    ):
        """Calculate MSE of magnitudes in Fourier space.

        Parameters
        ----------
        y_hat_batch
            Batched prediction.
        y_batch
            Batched target.
        weight_map_batch
            Optional weight map.
        """
        dim = tuple(range(1, len(y_batch.size())))
        
        y_hat_fft_mag_batch = torch.fft.rfftn(y_hat_batch, dim=dim).abs()
        y_fft_mag_batch = torch.fft.rfftn(y_batch, dim=dim).abs()
        
        if weight_map_batch is None:
            return F.mse_loss(y_hat_fft_mag_batch, y_fft_mag_batch)
        dim = tuple(range(1, len(weight_map_batch.size())))
        return (weight_map_batch * (y_hat_batch - y_batch) ** 2).sum(dim=dim).mean()

    
class SpectralMSE(torch.nn.Module):
    """Spectral and pixel MSE combined loss."""

    def forward(
        self,
        y_hat_batch: torch.Tensor,
        y_batch: torch.Tensor,
        weight_map_batch: Optional[torch.Tensor] = None,
        alpha: float = 0.2
    ):
        """Calculate MSEs of images and of their magnitudes in Fourier space.

        Parameters
        ----------
        y_hat_batch
            Batched prediction.
        y_batch
            Batched target.
        weight_map_batch
            Optional weight map.
        alpha
            Weight of spectral loss in the comination with MSE.
        """
        dim = tuple(range(1, len(y_batch.size())))
        
        y_hat_fft_mag_batch = torch.fft.rfftn(y_hat_batch, dim=dim).abs()
        y_fft_mag_batch = torch.fft.rfftn(y_batch, dim=dim).abs()
        spectral_loss = F.mse_loss(y_hat_fft_mag_batch, y_fft_mag_batch)
        
        if weight_map_batch is None:
            mse_loss = F.mse_loss(y_hat_batch, y_batch)
        else:
            mse_loss = (weight_map_batch * (y_hat_batch - y_batch) ** 2).sum(dim=dim).mean()

        return (1 - alpha) * mse_loss + alpha * spectral_loss
    
    
class JaccardBCE(torch.nn.Module):
    """Segmentation loss based on patch thresholding."""

    def forward(
        self,
        y_hat_batch: torch.Tensor,
        y_batch: torch.Tensor,
        weight_map_batch: Optional[torch.Tensor] = None,
        threshold = 0.005,
        alpha: float = 0.5
    ):
        """Calculate loss defined as alpha * BCE - (1 - alpha) * log (SoftJaccard).

        Parameters
        ----------
        y_hat_batch
            Batched prediction.
        y_batch
            Batched target.
        weight_map_batch
            Optional weight map.
        threshold
            Threshold Value for binarizing intensity target images.
        alpha
            Weight of spectral loss in the comination with BCE.
        """
        eps = 1e-15
        
        bin_y_batch = (y_batch >= threshold).float()
        soft_y_hat_batch = torch.sigmoid(y_hat_batch)

        intersection = (soft_y_hat_batch * bin_y_batch).sum()
        union = soft_y_hat_batch.sum() + bin_y_batch.sum()
        soft_jaccard = intersection / (union - intersection + eps)
        
        bce = F.binary_cross_entropy_with_logits(y_hat_batch, bin_y_batch)
        
        return (1 - alpha) * bce - alpha * torch.log(soft_jaccard)

    
class JaccardMSE(torch.nn.Module):
    """Combination loss based on pixel-MSE and IoU with patch thresholding."""

    def forward(
        self,
        y_hat_batch: torch.Tensor,
        y_batch: torch.Tensor,
        weight_map_batch: Optional[torch.Tensor] = None,
        threshold = 0.005,
        alpha: float = 0.5
    ):
        """Calculate loss defined as alpha * MSE - (1 - alpha) * log (SoftJaccard).

        Parameters
        ----------
        y_hat_batch
            Batched prediction.
        y_batch
            Batched target.
        weight_map_batch
            Optional weight map.
        threshold
            Threshold Value for binarizing intensity target images.
        alpha
            Weight of spectral loss in the comination with MSE.
        """
        eps = 1e-15
        
        bin_y_batch = (y_batch >= threshold).float()
        soft_y_hat_batch = torch.sigmoid(y_hat_batch)

        intersection = (soft_y_hat_batch * bin_y_batch).sum()
        union = soft_y_hat_batch.sum() + bin_y_batch.sum()
        soft_jaccard = intersection / (union - intersection + eps)
        
        if weight_map_batch is None:
            mse_loss = F.mse_loss(y_hat_batch, y_batch)
        else:
            mse_loss = (weight_map_batch * (y_hat_batch - y_batch) ** 2).sum(dim=dim).mean()
        
        return (1 - alpha) * mse_loss - alpha * torch.log(soft_jaccard)

    
class JaccardSoftMSE(torch.nn.Module):
    """Combination loss based on pixel-MSE and IoU with patch thresholding."""

    def forward(
        self,
        y_hat_batch: torch.Tensor,
        y_batch: torch.Tensor,
        weight_map_batch: Optional[torch.Tensor] = None,
        threshold = 0.005,
        alpha: float = 0.5
    ):
        """Calculate loss defined as alpha * MSE - (1 - alpha) * log (SoftJaccard).

        Parameters
        ----------
        y_hat_batch
            Batched prediction.
        y_batch
            Batched target.
        weight_map_batch
            Optional weight map.
        threshold
            Threshold Value for binarizing intensity target images.
        alpha
            Weight of spectral loss in the comination with MSE.
        """
        eps = 1e-15

        bin_y_batch = (y_batch >= threshold).float()
        soft_y_hat_batch = torch.sigmoid(y_hat_batch)

        intersection = (soft_y_hat_batch * bin_y_batch).sum()
        union = soft_y_hat_batch.sum() + bin_y_batch.sum()
        soft_jaccard = intersection / (union - intersection + eps)

        if weight_map_batch is None:
            mse_loss = F.mse_loss(soft_y_hat_batch, y_batch)
        else:
            mse_loss = (weight_map_batch * (soft_y_hat_batch - y_batch) ** 2).sum(dim=dim).mean()

        return (1 - alpha) * mse_loss - alpha * torch.log(soft_jaccard)


class MTLLoss(torch.nn.Module):
    """Combination loss based on pixel-MSE and IoU with patch thresholding."""

    def forward(
        self,
        y_hat_batch_mtl, # tuple of tensors: (predicted pixel values, predicted probability map)
        y_batch: torch.Tensor,
        weight_map_batch: Optional[torch.Tensor] = None,
        threshold = 0.1,
        alpha: float = 0.5
    ):
        """Calculate loss defined as alpha * MSE - (1 - alpha) * log (SoftJaccard).

        Parameters
        ----------
        y_hat_batch
            Batched prediction.
        y_batch
            Batched target.
        weight_map_batch
            Optional weight map.
        threshold
            Threshold Value for binarizing intensity target images.
        alpha
            Weight of spectral loss in the comination with MSE.
        """
        eps = 1e-15

        y_hat_batch, bin_y_hat_batch = y_hat_batch_mtl

        # threshold input image
        bin_y_batch = (y_batch >= threshold).float()
        bin_y_hat_batch = torch.sigmoid(bin_y_hat_batch)

        intersection = (bin_y_hat_batch * bin_y_batch).sum()
        union = bin_y_hat_batch.sum() + bin_y_batch.sum()
        soft_jaccard = 1 - intersection / (union - intersection + eps)

        if weight_map_batch is None:
            mse_loss = F.mse_loss(y_hat_batch, y_batch)
        else:
            mse_loss = (weight_map_batch * (y_hat_batch - y_batch) ** 2).sum(dim=dim).mean()

        # segmentation only
        # return (1 - alpha) * mse_loss + alpha * soft_jaccard
        return soft_jaccard