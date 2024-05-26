"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
import numbers

from ...fastmri import data
from ...fastmri.fftc import fft2c_new, ifft2c_new
import torch
from torch import nn
from torch.nn import functional as F
from ... import fastmri
from einops import rearrange

class MDRNet(nn.Module):
    """
    PyTorch implementation of a MDRNet model.

    O. Ronneberger, P. Fischer, and Thomas Brox. U-net: Convolutional networks
    for biomedical image segmentation. In International Conference on Medical
    image computing and computer-assisted intervention, pages 234–241.
    Springer, 2015.
    """

    def __init__(
            self,
            in_chans: int,
            out_chans: int,
            chans: int = 32,
            num_pool_layers: int = 3,
            drop_prob: float = 0.0,
    ):
        """
        Args:
            in_chans: Number of channels in the input to the U-Net model.
            out_chans: Number of channels in the output to the U-Net model.
            chans: Number of output channels of the first convolution layer.
            num_pool_layers: Number of down-sampling and up-sampling layers.
            drop_prob: Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob

        self.down_sample_layers = nn.ModuleList([MDLBlock(chans, chans * 2, drop_prob)])

        ch = chans * 2
        for _ in range(num_pool_layers - 1):
            self.down_sample_layers.append(MDLBlock(ch, ch * 2, drop_prob))
            ch *= 2
        self.conv = MDLBlock(ch, ch * 2, drop_prob)

        self.up_conv = nn.ModuleList()
        self.up_transpose_conv = nn.ModuleList()
        self.down_dimension = nn.ModuleList()
        for _ in range(num_pool_layers - 1):
            self.up_transpose_conv.append(TransposeConvBlock(ch * 2, ch))
            self.down_dimension.append(ConvBlock_DownF(ch * 2, ch // 2, drop_prob))
            self.up_conv.append(MDLBlock(ch // 2, ch, drop_prob))
            ch //= 2

        self.up_transpose_conv.append(TransposeConvBlock(ch * 2, ch))
        self.down_dimension.append(ConvBlock_DownF(ch * 2, ch // 2, drop_prob))

        self.up_conv.append(
            nn.Sequential(
                ConvBlock(ch // 2, ch, drop_prob),
                nn.Conv2d(ch, self.out_chans, kernel_size=1, stride=1),
            )
        )

        self.preconv = nn.Sequential(
            nn.Conv2d(in_chans, 32, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(drop_prob),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.

        Returns:
            Output tensor of shape `(N, out_chans, H, W)`.
        """
        stack = []
        output = self.preconv(image)

        # apply down-sampling layers
        for layer in self.down_sample_layers:
            output = layer(output)
            stack.append(output)
            output = F.avg_pool2d(output, kernel_size=2, stride=2, padding=0)

        output = self.conv(output)

        # apply up-sampling layers
        for transpose_conv, conv, down_dim in zip(self.up_transpose_conv, self.up_conv, self.down_dimension):
            downsample_layer = stack.pop()
            output = transpose_conv(output)

            # reflect pad on the right/botton if needed to handle odd input dimensions
            padding = [0, 0, 0, 0]
            if output.shape[-1] != downsample_layer.shape[-1]:
                padding[1] = 1  # padding right
            if output.shape[-2] != downsample_layer.shape[-2]:
                padding[3] = 1  # padding bottom
            if torch.sum(torch.tensor(padding)) != 0:
                output = F.pad(output, padding, "reflect")

            output = torch.cat([output, downsample_layer], dim=1)
            output = down_dim(output)
            output = conv(output)

        return output




class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),  # 第一个全连接层 (降维)
            nn.ReLU(inplace=True),  # ReLU 非线性激活函数
            nn.Linear(channel // reduction, channel, bias=False),  # 第二个全连接层 (升维)
            nn.Sigmoid()  # 非线性激活函数 + 数值范围约束 (0, 1)
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)  # 即上文所述的 U
        y = self.fc(y).view(b, c, 1, 1)  # reshape 张量以便于进行通道重要性加权的乘法操作

        return x * y.expand_as(x)  # 按元素一一对应相乘



class ConvBlock(nn.Module):
    """
    A Convolutional Block that consists of two convolution layers each followed by
    instance normalization, LeakyReLU activation and dropout.
    """

    def __init__(self, in_chans: int, out_chans: int, drop_prob: float):
        """
        Args:
            in_chans: Number of channels in the input.
            out_chans: Number of channels in the output.
            drop_prob: Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.drop_prob = drop_prob

        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(drop_prob),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(drop_prob),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.

        Returns:
            Output tensor of shape `(N, out_chans, H, W)`.
        """
        return self.layers(image)


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = (x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

class MDLBlock(nn.Module):
    """
    A Convolutional Block that consists of two convolution layers each followed by
    instance normalization, LeakyReLU activation and dropout.
    """

    def __init__(self, in_chans: int, out_chans: int, drop_prob: float):
        """
        Args:
            in_chans: Number of channels in the input.
            out_chans: Number of channels in the output.
            drop_prob: Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.drop_prob = drop_prob

        self.layers1 = nn.Sequential(
            nn.Conv2d(in_chans, in_chans//2, kernel_size=1, stride=1, bias=False),
            nn.InstanceNorm2d(in_chans//2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(drop_prob),
            nn.Conv2d(in_chans//2, in_chans // 2, kernel_size=3,groups=in_chans//2, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(in_chans // 2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(drop_prob),
        )
        self.layers2 = nn.Sequential(
            nn.Conv2d(int(in_chans //2), in_chans//2, kernel_size=3,groups=in_chans//2, padding=1,stride=1, bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(drop_prob),
            nn.Conv2d(in_chans//2, in_chans // 2, kernel_size=3,groups=in_chans//2, stride=1, padding=1, bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(drop_prob),
        )

        self.fu = nn.Sequential(
            nn.Conv2d(in_chans//2, in_chans // 2, kernel_size=1, stride=1, bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        self.fu2 = nn.Sequential(
            nn.Conv2d(in_chans // 2, in_chans // 2, kernel_size=1, stride=1, bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        self.fu3 = nn.Sequential(
            nn.Conv2d((in_chans ), in_chans//2 , kernel_size=1, stride=1, bias=False),

        )

        self.se_layer = SELayer(out_chans, 8)
        self.finlayer = nn.Sequential(
            nn.Conv2d(out_chans, out_chans, kernel_size=1, stride=1, bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(drop_prob),
        )

        self.se_layer_pre = SELayer(int(in_chans * (3 / 2)), 8)
        self.finlayer_pre = nn.Sequential(
            nn.Conv2d(int(in_chans * (3 / 2)), int(in_chans * (3 / 2)), kernel_size=1, stride=1, bias=False),
            nn.InstanceNorm2d(int(in_chans * (3 / 2))),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(drop_prob),
        )
        self.conv3 = nn.Sequential(nn.Conv2d(int(in_chans * (1 / 2)), int(in_chans * (1 / 2)), kernel_size=3,
                                       groups=int(in_chans * (1 / 2)), padding=1, stride=1, bias=False),)
        self.norm1 = LayerNorm(int(out_chans ), 'WithBias')
        self.attn = Attention(int(out_chans), 1, bias=False)
        self.norm2 = LayerNorm(int(in_chans * (3 / 2)), 'WithBias')

        self.ffn = FeedForward(int(in_chans * (3 / 2)), 1, bias=False)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.

        Returns:
            Output tensor of shape `(N, out_chans, H, W)`.
        """

        l1 = self.layers1(image)

        h = l1.shape[2]
        w = l1.shape[3]

        image_ = self.fu3(image)
        kspace = fastmri.fft2c(image_.view(-1, h, w, 2)).view(1, -1, h, w)

        l2_pre = self.fu(kspace)
        l2_pre_2 = self.fu2(l2_pre)
        l2_pre = l2_pre_2 + kspace

        l2 = fastmri.ifft2c(l2_pre.view(-1, h, w, 2)).view(1, -1, h, w)

        l1 = l1+self.conv3(l2)
        l2 = l2+self.conv3(l1)

        l1 = l1+self.layers2(l1)
        kspace = fastmri.fft2c(l2.view(-1, h, w, 2)).view(1, -1, h, w)

        l2_pre = self.fu(kspace)
        l2_pre_2 = self.fu2(l2_pre)
        l2_pre = l2_pre_2 + kspace

        l2 = fastmri.ifft2c(l2_pre.view(-1, h, w, 2)).view(1, -1, h, w)



        output = torch.cat([image, l1,l2], dim=1)



        output = output + self.attn(self.norm1(output))

        return output






class ConvBlock_DownF(nn.Module):
    """
    A Convolutional Block that consists of two convolution layers each followed by
    instance normalization, LeakyReLU activation and dropout.
    """

    def __init__(self, in_chans: int, out_chans: int, drop_prob: float):
        """
        Args:
            in_chans: Number of channels in the input.
            out_chans: Number of channels in the output.
            drop_prob: Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.drop_prob = drop_prob

        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, in_chans // 4, kernel_size=1, stride=1, bias=False),
            nn.InstanceNorm2d(in_chans // 4),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(drop_prob),

        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.

        Returns:
            Output tensor of shape `(N, out_chans, H, W)`.
        """
        return self.layers(image)


class TransposeConvBlock(nn.Module):
    """
    A Transpose Convolutional Block that consists of one convolution transpose
    layers followed by instance normalization and LeakyReLU activation.
    """

    def __init__(self, in_chans: int, out_chans: int):
        """
        Args:
            in_chans: Number of channels in the input.
            out_chans: Number of channels in the output.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans

        self.layers = nn.Sequential(
            nn.ConvTranspose2d(
                in_chans, out_chans, kernel_size=2, stride=2, bias=False
            ),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.

        Returns:
            Output tensor of shape `(N, out_chans, H*2, W*2)`.
        """
        return self.layers(image)
