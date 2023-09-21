"""Loss functions for fnet models."""
from typing import Optional

import tifffile
import numpy as np

import torch
from torch.nn import functional as F

from pytorch3dunet.unet3d.losses import DiceLoss, BCEDiceLoss

from fft_conv_pytorch import fft_conv


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
        loss_batch = 0.5 * torch.exp(-log_var_batch) * (mean_batch - y_batch).pow(2) + 0.5 * log_var_batch
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
        delta: float = 1.0,
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
            return (delta**2 * (torch.sqrt(1 + ((y_hat_batch - y_batch) / delta) ** 2) - 1)).mean()
        dim = tuple(range(1, len(weight_map_batch.size())))
        # weight map should sum up to 1.0 across all pixels in each image
        return (
            (weight_map_batch * (delta**2 * (torch.sqrt(1 + ((y_hat_batch - y_batch) / delta) ** 2) - 1)))
            .sum(dim=dim)
            .mean()
        )


class SpectralLoss(torch.nn.Module):
    """Spectral loss."""

    def forward(
        self, y_hat_batch: torch.Tensor, y_batch: torch.Tensor, weight_map_batch: Optional[torch.Tensor] = None
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
        alpha: float = 0.2,
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
        threshold=0.005,
        alpha: float = 0.5,
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
        threshold=0.005,
        alpha: float = 0.5,
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
            dim = tuple(range(1, len(weight_map_batch.size())))
            mse_loss = (weight_map_batch * (y_hat_batch - y_batch) ** 2).sum(dim=dim).mean()

        return (1 - alpha) * mse_loss - alpha * torch.log(soft_jaccard)


class JaccardSoftMSE(torch.nn.Module):
    """Combination loss based on pixel-MSE and IoU with patch thresholding."""

    def forward(
        self,
        y_hat_batch: torch.Tensor,
        y_batch: torch.Tensor,
        weight_map_batch: Optional[torch.Tensor] = None,
        threshold=0.005,
        alpha: float = 0.5,
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
            dim = tuple(range(1, len(weight_map_batch.size())))
            mse_loss = (weight_map_batch * (soft_y_hat_batch - y_batch) ** 2).sum(dim=dim).mean()

        return (1 - alpha) * mse_loss - alpha * torch.log(soft_jaccard)


class MSEBCEDiceLoss(torch.nn.Module):
    """Linear combination of MSE, BCE, and Dice losses"""

    def __init__(self, alpha, beta, gamma):
        super(MSEBCEDiceLoss, self).__init__()
        self.gamma = gamma
        self.bcedice = BCEDiceLoss(alpha=alpha, beta=beta)

    def forward(
        self,
        y_hat_batch: torch.Tensor,
        y_batch: torch.Tensor,
        weight_map_batch: torch.Tensor,
    ):
        mse_loss = F.mse_loss(y_hat_batch, y_batch)
        bce_dice_loss = self.bcedice(y_hat_batch, (weight_map_batch > 0).float())

        return (1 - self.gamma) * bce_dice_loss + self.gamma * mse_loss


class WMSEDiceLoss(torch.nn.Module):
    """Linear combination of MSE and Dice losses."""

    def __init__(self, gamma):
        super(WMSEDiceLoss, self).__init__()
        self.gamma = gamma
        self.dice = DiceLoss(normalization="none")

    def tune_sigm(self, x, k=-0.95):
        denominator = k - 2 * k * torch.abs(x) + 1
        return (x - k * x) / denominator.clamp(min=torch.finfo(torch.float32).eps)

    def forward(
        self,
        y_hat_batch: torch.Tensor,
        y_batch: torch.Tensor,
        weight_map_batch: Optional[torch.Tensor] = None,
    ):
        # # unweighted MSE option (mse_dice_tunesigm_bin)
        # mse_loss = F.mse_loss(y_hat_batch, y_batch)

        if weight_map_batch is None:
            mse_loss = F.mse_loss(y_hat_batch, y_batch)
            dice_loss = self.dice(self.tune_sigm(y_hat_batch), self.tune_sigm(y_batch))
        else:
            # there is a bug in preprocessing that converts [0,1] weight map to [0,257]
            weight_map_batch = (weight_map_batch > 0).float()
            mse_loss = (weight_map_batch * (y_hat_batch - y_batch) ** 2).mean()
            dice_loss = self.dice(self.tune_sigm(y_hat_batch), weight_map_batch)

        return (1 - self.gamma) * dice_loss + self.gamma * mse_loss


class PSFMSE(torch.nn.Module):
    """Loss with PSF."""

    def __init__(self, psf_path: str):
        """Initialize loss module.

        Parameters
        ----------
        psf_path
            Path to the point spread function.
        """
        super(PSFMSE, self).__init__()

        psf = tifffile.imread(psf_path)

        # calculate padding
        padding = ((psf.shape[0] - 1) // 2, (psf.shape[1] - 1) // 2, (psf.shape[2] - 1) // 2)
        self.padding = tuple(x for x in reversed(padding) for _ in range(2))

        # normalize psf
        psf = torch.tensor(psf[None, None, :, :, :], dtype=torch.float32).cuda()
        self.psf = psf / psf.sum()

    def forward(
        self,
        y_hat_batch: torch.Tensor,
        y_batch: torch.Tensor,
        weight_map_batch: Optional[torch.Tensor] = None,
    ):
        """Calculate loss after convolving PSF.

        Parameters
        ----------
        y_hat_batch
            Batched prediction.
        y_batch
            Batched target.
        weight_map_batch
            Optional weight map.
        """
        # convolve with PSF
        y_hat_batch = fft_conv(F.pad(y_hat_batch, self.padding, "reflect"), self.psf)

        # TODO: exclude border voxels from loss calculation
        if weight_map_batch is None:
            mse_loss = F.mse_loss(y_hat_batch, y_batch)
        else:
            dim = tuple(range(1, len(weight_map_batch.size())))
            mse_loss = (weight_map_batch * (y_hat_batch - y_batch) ** 2).sum(dim=dim).mean()

        return mse_loss


class PSFWMSEDiceLoss(torch.nn.Module):
    """Linear combination of MSE and Dice losses with PSF."""

    def __init__(self, gamma: float, psf_path: str):
        """Initialize loss module.

        Parameters
        ----------
        gamma
            Weight of MSE loss.
        psf_path
            Path to the point spread function.
        """
        super(PSFWMSEDiceLoss, self).__init__()
        self.gamma = gamma
        self.dice = DiceLoss(normalization="none")

        psf = tifffile.imread(psf_path)
        # calculate padding
        padding = ((psf.shape[0] - 1) // 2, (psf.shape[1] - 1) // 2, (psf.shape[2] - 1) // 2)
        self.padding = tuple(x for x in reversed(padding) for _ in range(2))
        # normalize psf
        psf = torch.tensor(psf[None, None, :, :, :], dtype=torch.float32).cuda()
        self.psf = psf / psf.sum()

    def tune_sigm(self, x, k=-0.95):
        denominator = k - 2 * k * torch.abs(x) + 1
        return (x - k * x) / denominator.clamp(min=torch.finfo(torch.float32).eps)

    def forward(
        self,
        y_hat_batch: torch.Tensor,
        y_batch: torch.Tensor,
        weight_map_batch: Optional[torch.Tensor] = None,
    ):
        """Calculate loss after convolving PSF.

        Parameters
        ----------
        y_hat_batch
            Batched prediction.
        y_batch
            Batched target.
        weight_map_batch
            Optional weight map.
        """
        # convolve with PSF
        y_hat_batch = fft_conv(F.pad(y_hat_batch, self.padding, "reflect"), self.psf)

        if weight_map_batch is None:
            mse_loss = F.mse_loss(y_hat_batch, y_batch)
            dice_loss = self.dice(self.tune_sigm(y_hat_batch), self.tune_sigm(y_batch))
        else:
            # there is a bug in preprocessing that converts [0,1] weight map to [0,257]
            weight_map_batch = (weight_map_batch > 0).float()
            mse_loss = (weight_map_batch * (y_hat_batch - y_batch) ** 2).mean()
            dice_loss = self.dice(self.tune_sigm(y_hat_batch), weight_map_batch)

        return (1 - self.gamma) * dice_loss + self.gamma * mse_loss


class PSNRLoss(torch.nn.Module):
    def __init__(self, loss_weight=1.0):
        super(PSNRLoss, self).__init__()
        self.loss_weight = loss_weight
        self.scale = 10 / np.log(10)

    def forward(self, pred, target):
        assert len(pred.size()) == 5 and pred.size(1) == 1
        assert len(target.size()) == 5 and target.size(1) == 1

        mse = ((pred - target) ** 2).mean(dim=(1, 2, 3, 4))
        psnr_loss = self.scale * torch.log(mse + 1e-8)
        return self.loss_weight * psnr_loss.mean()


class FRC3DLoss(torch.nn.Module):
    def __init__(self, loss_weight=1.0):
        super(FRC3DLoss, self).__init__()

    def radial_mask_3d(self, r, cx=32, cy=64, cz=64, delta=1):
        sz, sy, sx = torch.meshgrid(torch.arange(0, cz), torch.arange(0, cy), torch.arange(0, cx))
        ind = (sx - cx // 2)**2 + (sy - cy // 2)**2 + (sz - cz // 2)**2
        ind1 = ind <= ((r + delta)**2)
        ind2 = ind > (r**2)
        return ind1 * ind2


    def get_radial_masks_3d(self, image_shape=(32, 64, 64)):
        freq_nyq = torch.tensor([image_shape[0] // 2, image_shape[1] // 2, image_shape[2] // 2]).float()
        radii = torch.arange(1, freq_nyq.min().int() + 1).reshape(-1, 1)

        radial_masks = [self.radial_mask_3d(radius.item(), cx=image_shape[0], cy=image_shape[1], cz=image_shape[2]) for radius in radii]
        radial_masks = torch.stack(radial_masks, axis=0)  # Shape: (num_radii, cz, cy, cx)
        radial_masks = radial_masks.permute(0, 3, 2, 1)  # Shape: (num_radii, cx, cy, cz)
        radial_masks = radial_masks.unsqueeze(1)  # Shape: (num_radii, 1, cx, cy, cz)

        spatial_freq = radii.float() / freq_nyq.min()
        spatial_freq = spatial_freq / spatial_freq.max()

        return radial_masks, spatial_freq


    def fft_rn_img(self, img, rn):
        img = img.to(dtype=torch.complex64)
        fft_img = torch.fft.fftn(img, dim=[-3, -2, -1])
        fft_img = torch.fft.fftshift(fft_img, dim=[-3, -2, -1])
        return fft_img.unsqueeze(1) * rn.unsqueeze(0)
    

    def fourier_ring_correlation(self, image1, image2, rn, spatial_freq):
        rn = rn.to(torch.device('cuda'), dtype=torch.complex64)
        t1 = self.fft_rn_img(image1, rn)
        t2 = self.fft_rn_img(image2, rn)

        c1 = torch.real(torch.sum(t1 * torch.conj(t2), dim=[-3, -2, -1]))
        c2 = torch.sum(torch.abs(t1) ** 2, dim=[-3, -2, -1])
        c3 = torch.sum(torch.abs(t2) ** 2, dim=[-3, -2, -1])

        frc = c1 / torch.sqrt(c2 * c3)        
        frc[torch.isinf(frc)] = 0
        frc[torch.isnan(frc)] = 0

        t = spatial_freq.cuda()
        y = frc.mean(0)
        riemann_sum = torch.sum((t[1:] - t[:-1]) * (y[:-1] + y[1:]) / 2.0, dim=0)
        return riemann_sum


    def forward(self, pred, target):
        assert len(pred.size()) == 5 and pred.size(1) == 1
        assert len(target.size()) == 5 and target.size(1) == 1

        radial_masks, spatial_freq = self.get_radial_masks_3d(image_shape=pred.shape[-3:])
        frc = self.fourier_ring_correlation(pred, target, radial_masks, spatial_freq)
        return 1 - frc
