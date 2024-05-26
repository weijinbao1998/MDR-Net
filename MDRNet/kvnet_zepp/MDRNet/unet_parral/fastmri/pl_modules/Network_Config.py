"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from argparse import ArgumentParser
import matplotlib.pyplot as plt

import torch
from torch import nn
import MDRNet
from torch.nn import functional as F

from ... import fastmri
from ...fastmri.data import transforms
from ...fastmri.losses import SSIMLoss



from .mri_module import MriModule
from ..losses import FocalFrequencyLoss


class Network(nn.Module):
    def __init__(self, in_chans, out_chans, chans, num_pools=3):
        super().__init__()
        self.net = MDRNet(in_chans, out_chans, chans, num_pools)

    def norm(self, x):
        # Group norm
        b, c, h, w = x.shape
        x = x.contiguous().view(b, 2, c // 2 * h * w)
        mean = x.mean(dim=2).view(b, 2, 1, 1, 1).expand(
            b, 2, c // 2, 1, 1).contiguous().view(b, c, 1, 1)
        std = x.std(dim=2).view(b, 2, 1, 1, 1).expand(
            b, 2, c // 2, 1, 1).contiguous().view(b, c, 1, 1)
        x = x.view(b, c, h, w)
        return (x - mean) / std, mean, std

    def unnorm(self, x, mean, std):
        return x * std + mean

    def forward(self, x):
        x, mean_org, std_org = self.norm(x)
        z = self.net(x)

        z = self.unnorm(z, mean_org, std_org)
        return z



class MDRNetSimpleModule(MriModule):
    """
    Unet training module.

    This can be used to train baseline U-Nets from the paper:

    J. Zbontar et al. fastMRI: An Open Dataset and Benchmarks for Accelerated
    MRI. arXiv:1811.08839. 2018.
    """

    def __init__(
            self,
            in_chans=1,
            out_chans=1,
            chans=32,
            num_pool_layers=3,
            drop_prob=0.0,
            lr=0.001,
            lr_step_size=40,
            lr_gamma=0.1,
            weight_decay=0.0,
            **kwargs,
    ):
        """
        Args:
            in_chans (int, optional): Number of channels in the input to the
                U-Net model. Defaults to 1.
            out_chans (int, optional): Number of channels in the output to the
                U-Net model. Defaults to 1.
            chans (int, optional): Number of output channels of the first
                convolution layer. Defaults to 32.
            num_pool_layers (int, optional): Number of down-sampling and
                up-sampling layers. Defaults to 4.
            drop_prob (float, optional): Dropout probability. Defaults to 0.0.
            lr (float, optional): Learning rate. Defaults to 0.001.
            lr_step_size (int, optional): Learning rate step size. Defaults to
                40.
            lr_gamma (float, optional): Learning rate gamma decay. Defaults to
                0.1.
            weight_decay (float, optional): Parameter for penalizing weights
                norm. Defaults to 0.0.
        """
        super().__init__(**kwargs)
        self.save_hyperparameters()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob
        self.lr = lr
        self.lr_step_size = lr_step_size
        self.lr_gamma = lr_gamma
        self.weight_decay = weight_decay
        self.dtrans_i = nn.ModuleList()
        self.dtrans_k = nn.ModuleList()


        self.dc_weight_i = nn.ParameterList()
        for i in range(4):
            self.dc_weight_i.append(nn.Parameter(torch.ones(1)))
            self.dtrans_i.append(Network(2, 2, 32, 3))

        self.loss = SSIMLoss()
        self.frq_loss = FocalFrequencyLoss(loss_weight=0.2,
                                           alpha=2.0,
                                           patch_factor=2,
                                           ave_spectrum=True,
                                           log_matrix=True,
                                           batch_matrix=True,
                                           ).cuda()

    def forward(self, image, mask, masked_kspace, image_org):
        kspace = masked_kspace.clone().permute(0, 3, 1, 2)
        zero = torch.zeros(1, 1, 1, 1).to(masked_kspace)
        image = image_org

        for li, wi in zip(self.dtrans_i, self.dc_weight_i):
            image = li(image) + image_org
            image_k = fastmri.fft2c(image.permute(0, 2, 3, 1))
            image_k_dc = image_k - torch.where(mask, image_k - masked_kspace, zero) * wi
            image = fastmri.ifft2c(image_k_dc).permute(0, 3, 1, 2)

        unet_out_abs = fastmri.complex_abs(image.permute(0, 2, 3, 1))
        return unet_out_abs

    def training_step(self, batch, batch_idx):
        image, target, mean, std, fname, slice_num, max_value, mask, masked_kspace, image_org = batch
        output = self(image, mask, masked_kspace, image_org)
        output, target = transforms.center_crop_to_smallest(output.unsqueeze(0), target.unsqueeze(0))
        loss = self.loss(output, target, max_value) + self.frq_loss

        self.log("loss", loss.detach())

        return loss

    def validation_step(self, batch, batch_idx):
        image, target, mean, std, fname, slice_num, max_value, mask, masked_kspace, image_org = batch
        output = self(image, mask, masked_kspace, image_org)
        output, target = transforms.center_crop_to_smallest(output.unsqueeze(0), target.unsqueeze(0))


        return {
            "batch_idx": batch_idx,
            "fname": fname,
            "slice_num": slice_num,
            "max_value": max_value,
            "output": output[0][0],
            "target": target[0][0],
            "val_loss": self.loss(output, target, max_value) + self.frq_loss,
        }

    def test_step(self, batch, batch_idx):
        image, target, mean, std, fname, slice_num, max_value, mask, masked_kspace, image_org = batch
        output = self(image, mask, masked_kspace, image_org)

        output = transforms.center_crop(output, [320, 320])

        return {
            "fname": fname,
            "slice": slice_num,
            "output": output.cpu().numpy(),
        }

    def configure_optimizers(self):
        optim = torch.optim.RMSprop(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optim, self.lr_step_size, self.lr_gamma
        )

        return [optim], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no-cover
        """
        Define parameters that only apply to this model
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser = MriModule.add_model_specific_args(parser)

        # network params
        parser.add_argument(
            "--in_chans", default=1, type=int, help="Number of U-Net input channels"
        )
        parser.add_argument(
            "--out_chans", default=1, type=int, help="Number of U-Net output chanenls"
        )
        parser.add_argument(
            "--chans", default=1, type=int, help="Number of top-level U-Net filters."
        )
        parser.add_argument(
            "--num_pool_layers",
            default=4,
            type=int,
            help="Number of U-Net pooling layers.",
        )
        parser.add_argument(
            "--drop_prob", default=0.0, type=float, help="U-Net dropout probability"
        )

        # training params (opt)
        parser.add_argument(
            "--lr", default=0.001, type=float, help="RMSProp learning rate"
        )
        parser.add_argument(
            "--lr_step_size",
            default=40,
            type=int,
            help="Epoch at which to decrease step size",
        )
        parser.add_argument(
            "--lr_gamma", default=0.1, type=float, help="Amount to decrease step size"
        )
        parser.add_argument(
            "--weight_decay",
            default=0.0,
            type=float,
            help="Strength of weight decay regularization",
        )

        return parser
