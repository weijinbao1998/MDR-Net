"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SSIMLoss(nn.Module):
    """
    SSIM loss module.
    """

    def __init__(self, win_size: int = 7, k1: float = 0.01, k2: float = 0.03):
        """
        Args:
            win_size: Window size for SSIM calculation.
            k1: k1 parameter for SSIM calculation.
            k2: k2 parameter for SSIM calculation.
        """
        super().__init__()
        self.win_size = win_size
        self.k1, self.k2 = k1, k2
        self.register_buffer("w", torch.ones(1, 1, win_size, win_size) / win_size ** 2)
        NP = win_size ** 2
        self.cov_norm = NP / (NP - 1)

    def forward(self, X: torch.Tensor, Y: torch.Tensor, data_range: torch.Tensor):
        assert isinstance(self.w, torch.Tensor)

        data_range = data_range[:, None, None, None]
        C1 = (self.k1 * data_range) ** 2
        C2 = (self.k2 * data_range) ** 2
        ux = F.conv2d(X, self.w)  # typing: ignore
        uy = F.conv2d(Y, self.w)  #
        uxx = F.conv2d(X * X, self.w)
        uyy = F.conv2d(Y * Y, self.w)
        uxy = F.conv2d(X * Y, self.w)
        vx = self.cov_norm * (uxx - ux * ux)
        vy = self.cov_norm * (uyy - uy * uy)
        vxy = self.cov_norm * (uxy - ux * uy)
        A1, A2, B1, B2 = (
            2 * ux * uy + C1,
            2 * vxy + C2,
            ux ** 2 + uy ** 2 + C1,
            vx + vy + C2,
        )
        D = B1 * B2
        S = (A1 * A2) / D

        return 1 - S.mean()


class PSNRLoss(nn.Module):
    """
    SSIM loss module.
    """

    def __init__(self):
        """
        Args:
            win_size: Window size for SSIM calculation.
            k1: k1 parameter for SSIM calculation.
            k2: k2 parameter for SSIM calculation.
        """
        super().__init__()

    def forward(self, output: torch.Tensor, target: torch.Tensor):
        fenzi = torch.sum((target - output) ** 2)
        fenmu = torch.sum(target ** 2)
        loss = torch.log(fenzi / fenmu + 1) * 10
        # print('')
        return loss


class LogLoss(nn.Module):
    """
    SSIM loss module.
    """

    def __init__(self):
        """
        Args:
            win_size: Window size for SSIM calculation.
            k1: k1 parameter for SSIM calculation.
            k2: k2 parameter for SSIM calculation.
        """
        super().__init__()

    def forward(self, output: torch.Tensor, target: torch.Tensor):
        loss = torch.sum(torch.log(torch.abs(target - output) + 1)) / 320 / 2 / 240
        return loss


class GuiyiLoss(nn.Module):
    """
    SSIM loss module.
    """

    def __init__(self):
        """
        Args:
            win_size: Window size for SSIM calculation.
            k1: k1 parameter for SSIM calculation.
            k2: k2 parameter for SSIM calculation.
        """
        super().__init__()

    def forward(self, output: torch.Tensor, target: torch.Tensor):
        loss = torch.sum(((target - output) ** 2) / (target ** 2 + 0.0000001)) / 320 / 2 / 240
        return loss


class Guiyi1Loss(nn.Module):
    """
    SSIM loss module.
    """

    def __init__(self):
        """
        Args:
            win_size: Window size for SSIM calculation.
            k1: k1 parameter for SSIM calculation.
            k2: k2 parameter for SSIM calculation.
        """
        super().__init__()

    def forward(self, output: torch.Tensor, target: torch.Tensor):
        loss = torch.sum(torch.abs(target - output) / (torch.abs(target) + 0.0000001)) / 320 / 2 / 240
        return loss


class WeightLoss(nn.Module):
    """
    SSIM loss module.
    """

    def __init__(self):
        """
        Args:
            win_size: Window size for SSIM calculation.
            k1: k1 parameter for SSIM calculation.
            k2: k2 parameter for SSIM calculation.
        """
        super().__init__()
        self.weight = [1, 1, 1, 1]
        L_wid = 40
        M_wid = 40
        H_wid = 40
        R_wid = 160 - L_wid - M_wid - H_wid
        # L
        self.L = torch.zeros(1,320,320,2)
        self.L[:,160-L_wid:160+L_wid,160-L_wid:160+L_wid,:] = 1
        # M
        self.M = torch.zeros(1,320,320,2)
        self.M[:,160-(M_wid + L_wid):160+(M_wid + L_wid),160-(M_wid + L_wid):160+(M_wid + L_wid),:] = 1
        self.M[:,160-L_wid:160+L_wid,160-L_wid:160+L_wid,:] = 0
        # H
        self.H = torch.zeros(1,320,320,2)
        self.H[:,160-(M_wid + L_wid + H_wid):160+(M_wid + L_wid + H_wid),160-(M_wid + L_wid + H_wid):160+(M_wid + L_wid + H_wid),:] = 1
        self.H[:,160-(M_wid + L_wid):160+(M_wid + L_wid),160-(M_wid + L_wid):160+(M_wid + L_wid),:] = 0
        # T 
        self.T = torch.ones(1,320,320,2)
        self.T[:,160-(M_wid + L_wid + H_wid):160+(M_wid + L_wid + H_wid),160-(M_wid + L_wid + H_wid):160+(M_wid + L_wid + H_wid),:] = 0

    def forward(self, output: torch.Tensor, target: torch.Tensor):
        # junzhi
        L_mean = torch.sum(target * self.L) / torch.sum(self.L)
        M_mean = torch.sum(target * self.M) / torch.sum(self.M)
        H_mean = torch.sum(target * self.H) / torch.sum(self.H)
        T_mean = torch.sum(target * self.T) / torch.sum(self.T)
        s_L = torch.sum((target - L_mean) ** 2 * self.L)
        s_M = torch.sum((target - M_mean) ** 2 * self.M)
        s_H = torch.sum((target - H_mean) ** 2 * self.H)
        s_T = torch.sum((target - T_mean) ** 2 * self.T)
        loss_all = (target - output) ** 2
        loss_L  = torch.sum(loss_all * self.L) / s_L
        loss_M  = torch.sum(loss_all * self.M) / s_M
        loss_H  = torch.sum(loss_all * self.H) / s_H
        loss_T  = torch.sum(loss_all * self.T) / s_T
        return self.weight[0] * loss_L + self.weight[1] * loss_M + self.weight[2] * loss_H + self.weight[3] * loss_T


class l2_Loss(nn.Module):
    """
    SSIM loss module.
    """

    def __init__(self):
        """
        Args:
            win_size: Window size for SSIM calculation.
            k1: k1 parameter for SSIM calculation.
            k2: k2 parameter for SSIM calculation.
        """
        super().__init__()

    def forward(self, output: torch.Tensor, target: torch.Tensor):
        batch = output.shape[0]
        loss = torch.sum((target - output) ** 2) / batch / 64 / 64
        return loss

class FocalFrequencyLoss(nn.Module):
    """The torch.nn.Module class that implements focal frequency loss - a
    frequency domain loss function for optimizing generative models.
    Ref:
    Focal Frequency Loss for Image Reconstruction and Synthesis. In ICCV 2021.
    <https://arxiv.org/pdf/2012.12821.pdf>
    Args:
        loss_weight (float): weight for focal frequency loss. Default: 1.0
        alpha (float): the scaling factor alpha of the spectrum weight matrix for flexibility. Default: 1.0
        patch_factor (int): the factor to crop image patches for patch-based focal frequency loss. Default: 1
        ave_spectrum (bool): whether to use minibatch average spectrum. Default: False
        log_matrix (bool): whether to adjust the spectrum weight matrix by logarithm. Default: False
        batch_matrix (bool): whether to calculate the spectrum weight matrix using batch-based statistics. Default: False
    """


    def __init__(self, loss_weight=1.0, alpha=1.0, patch_factor=1, ave_spectrum=False, log_matrix=False, batch_matrix=True):
        super(FocalFrequencyLoss, self).__init__()
        self.loss_weight = loss_weight
        self.alpha = alpha
        self.patch_factor = patch_factor
        self.ave_spectrum = ave_spectrum
        self.log_matrix = log_matrix
        self.batch_matrix = batch_matrix

    def tensor2freq(self, x):
        # crop image patches
        patch_factor = self.patch_factor
        _, _, h, w = x.shape
        assert h % patch_factor == 0 and w % patch_factor == 0, (
            'Patch factor should be divisible by image height and width')
        patch_list = []
        patch_h = h // patch_factor
        patch_w = w // patch_factor
        for i in range(patch_factor):
            for j in range(patch_factor):
                patch_list.append(x[:, :, i * patch_h:(i + 1) * patch_h, j * patch_w:(j + 1) * patch_w])

        # stack to patch tensor
        y = torch.stack(patch_list, 1)

        # perform 2D DFT (real-to-complex, orthonormalization)
        if IS_HIGH_VERSION:
            freq = torch.fft.fft2(x, norm='ortho')
            freq = torch.stack([freq.real, freq.imag], -1)
        else:
            freq = torch.rfft(y, 2, onesided=False, normalized=True)
        return freq

    def loss_formulation(self, recon_freq, real_freq, matrix=None):
        # spectrum weight matrix
        if matrix is not None:
            # if the matrix is predefined
            weight_matrix = matrix.detach()
        else:
            # if the matrix is calculated online: continuous, dynamic, based on current Euclidean distance
            matrix_tmp = (recon_freq - real_freq) ** 2
            matrix_tmp = torch.sqrt(matrix_tmp[..., 0] + matrix_tmp[..., 1]) ** self.alpha

            # whether to adjust the spectrum weight matrix by logarithm
            if self.log_matrix:
                matrix_tmp = torch.log(matrix_tmp + 1.0)

            # whether to calculate the spectrum weight matrix using batch-based statistics
            if self.batch_matrix:
                matrix_tmp = matrix_tmp / matrix_tmp.max()
            else:
                matrix_tmp = matrix_tmp / matrix_tmp.max(-1).values.max(-1).values[:, :, :, None, None]

            matrix_tmp[torch.isnan(matrix_tmp)] = 0.0
            matrix_tmp = torch.clamp(matrix_tmp, min=0.0, max=1.0)
            weight_matrix = matrix_tmp.clone().detach()

        assert weight_matrix.min().item() >= 0 and weight_matrix.max().item() <= 1, (
            'The values of spectrum weight matrix should be in the range [0, 1], '
            'but got Min: %.10f Max: %.10f' % (weight_matrix.min().item(), weight_matrix.max().item()))

        # frequency distance using (squared) Euclidean distance
        tmp = (recon_freq - real_freq) ** 2
        freq_distance = tmp[..., 0] + tmp[..., 1]

        # dynamic spectrum weighting (Hadamard product)
        loss = weight_matrix * freq_distance
        return torch.mean(loss)

    def forward(self, pred, target, matrix=None, **kwargs):
        """Forward function to calculate focal frequency loss.
        Args:
            pred (torch.Tensor): of shape (N, C, H, W). Predicted tensor.
            target (torch.Tensor): of shape (N, C, H, W). Target tensor.
            matrix (torch.Tensor, optional): Element-wise spectrum weight matrix.
                Default: None (If set to None: calculated online, dynamic).
        """
        pred = (pred - torch.min(pred))/(torch.max(pred)-torch.min(pred))
        target = (target - torch.min(target))/(torch.max(target)-torch.min(target))
        pred_freq = self.tensor2freq(pred)
        target_freq = self.tensor2freq(target)

        # whether to use minibatch average spectrum
        if self.ave_spectrum:
            pred_freq = torch.mean(pred_freq, 0, keepdim=True)
            target_freq = torch.mean(target_freq, 0, keepdim=True)

        # calculate focal frequency loss
        return self.loss_formulation(pred_freq, target_freq, matrix) * self.loss_weight