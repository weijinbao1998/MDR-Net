"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
import copy
from typing import Dict, Optional, Sequence, Tuple, Union

from matplotlib import pyplot as plt
from torch._C import dtype
from torch.nn.functional import instance_norm

import fastmri
import numpy as np
import torch
# import cv2
from torch.nn import functional as F
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import normalized_root_mse as nmse
from .subsample import MaskFunc
from fastmri.data.subsample import create_mask_for_mask_type
# from pygrappa import grappa
import pywt


# from pytorch_wavelets import DWTForward, DWTInverse


from typing import Dict, Optional, Sequence, Tuple, Union

from time import time
from tempfile import NamedTemporaryFile as NTF

import numpy as np
from skimage.util import view_as_windows


def grappa(
        kspace, calib, kernel_size=(5, 5), coil_axis=-1, lamda=0.01,
        memmap=False, memmap_filename='out.memmap', silent=True):
    '''GeneRalized Autocalibrating Partially Parallel Acquisitions.

    Parameters
    ----------
    kspace : array_like
        2D multi-coil k-space data to reconstruct from.  Make sure
        that the missing entries have exact zeros in them.
    calib : array_like
        Calibration data (fully sampled k-space).
    kernel_size : tuple, optional
        Size of the 2D GRAPPA kernel (kx, ky).
    coil_axis : int, optional
        Dimension holding coil data.  The other two dimensions should
        be image size: (sx, sy).
    lamda : float, optional
        Tikhonov regularization for the kernel calibration.
    memmap : bool, optional
        Store data in Numpy memmaps.  Use when datasets are too large
        to store in memory.
    memmap_filename : str, optional
        Name of memmap to store results in.  File is only saved if
        memmap=True.
    silent : bool, optional
        Suppress messages to user.

    Returns
    -------
    res : array_like
        k-space data where missing entries have been filled in.

    Notes
    -----
    Based on implementation of the GRAPPA algorithm [1]_ for 2D
    images.

    If memmap=True, the results will be written to memmap_filename
    and nothing is returned from the function.

    References
    ----------
    .. [1] Griswold, Mark A., et al. "Generalized autocalibrating
           partially parallel acquisitions (GRAPPA)." Magnetic
           Resonance in Medicine: An Official Journal of the
           International Society for Magnetic Resonance in Medicine
           47.6 (2002): 1202-1210.
    '''

    # Remember what shape the final reconstruction should be
    fin_shape = kspace.shape[:]

    # Put the coil dimension at the end
    kspace = np.moveaxis(kspace, coil_axis, -1)
    calib = np.moveaxis(calib, coil_axis, -1)

    # Quit early if there are no holes
    if np.sum((np.abs(kspace[..., 0]) == 0).flatten()) == 0:
        return np.moveaxis(kspace, -1, coil_axis)

    # Get shape of kernel
    kx, ky = kernel_size[:]
    kx2, ky2 = int(kx/2), int(ky/2)
    nc = calib.shape[-1]

    # When we apply weights, we need to select a window of data the
    # size of the kernel.  If the kernel size is odd, the window will
    # be symmetric about the target.  If it's even, then we have to
    # decide where the window lies in relation to the target.  Let's
    # arbitrarily decide that it will be right-sided, so we'll need
    # adjustment factors used as follows:
    #     S = kspace[xx-kx2:xx+kx2+adjx, yy-ky2:yy+ky2+adjy, :]
    # Where:
    #     xx, yy : location of target
    adjx = np.mod(kx, 2)
    adjy = np.mod(ky, 2)

    # Pad kspace data
    kspace = np.pad(  # pylint: disable=E1102
        kspace, ((kx2, kx2), (ky2, ky2), (0, 0)), mode='constant')
    calib = np.pad(  # pylint: disable=E1102
        calib, ((kx2, kx2), (ky2, ky2), (0, 0)), mode='constant')

    # Notice that all coils have same sampling pattern, so choose
    # the 0th one arbitrarily for the mask
    mask = np.ascontiguousarray(np.abs(kspace[..., 0]) > 0)

    # Store windows in temporary files so we don't overwhelm memory
    with NTF() as fP, NTF() as fA, NTF() as frecon:

        # Start the clock...
        t0 = time()

        # Get all overlapping patches from the mask
        P = np.memmap(fP, dtype=mask.dtype, mode='w+', shape=(
            mask.shape[0]-2*kx2, mask.shape[1]-2*ky2, 1, kx, ky))
        P = view_as_windows(mask, (kx, ky))
        Psh = P.shape[:]  # save shape for unflattening indices later
        P = P.reshape((-1, kx, ky))

        # Find the unique patches and associate them with indices
        P, iidx = np.unique(P, return_inverse=True, axis=0)

        # Filter out geometries that don't have a hole at the center.
        # These are all the kernel geometries we actually need to
        # compute weights for.
        validP = np.argwhere(~P[:, kx2, ky2]).squeeze()

        # We also want to ignore empty patches
        invalidP = np.argwhere(np.all(P == 0, axis=(1, 2)))
        validP = np.setdiff1d(validP, invalidP, assume_unique=True)

        # Make sure validP is iterable
        validP = np.atleast_1d(validP)

        # Give P back its coil dimension
        P = np.tile(P[..., None], (1, 1, 1, nc))

        if not silent:
            print('P took %g seconds!' % (time() - t0))
        t0 = time()

        # Get all overlapping patches of ACS
        try:
            A = np.memmap(fA, dtype=calib.dtype, mode='w+', shape=(
                calib.shape[0]-2*kx, calib.shape[1]-2*ky, 1, kx, ky, nc))
            A[:] = view_as_windows(
                calib, (kx, ky, nc)).reshape((-1, kx, ky, nc))
        except ValueError:
            A = view_as_windows(
                calib, (kx, ky, nc)).reshape((-1, kx, ky, nc))

        # Report on how long it took to construct windows
        if not silent:
            print('A took %g seconds' % (time() - t0))

        # Initialize recon array
        recon = np.memmap(
            frecon, dtype=kspace.dtype, mode='w+',
            shape=kspace.shape)

        # Train weights and apply them for each valid hole we have in
        # kspace data:
        t0 = time()
        for ii in validP:
            # Get the sources by masking all patches of the ACS and
            # get targets by taking the center of each patch. Source
            # and targets will have the following sizes:
            #     S : (# samples, N possible patches in ACS)
            #     T : (# coils, N possible patches in ACS)
            # Solve the equation for the weights:
            #     WS = T
            #     WSS^H = TS^H
            #  -> W = TS^H (SS^H)^-1
            # S = A[:, P[ii, ...]].T # transpose to get correct shape
            # T = A[:, kx2, ky2, :].T
            # TSh = T @ S.conj().T
            # SSh = S @ S.conj().T
            # W = TSh @ np.linalg.pinv(SSh) # inv won't work here

            # Equivalenty, we can formulate the problem so we avoid
            # computing the inverse, use numpy.linalg.solve, and
            # Tikhonov regularization for better conditioning:
            #     SW = T
            #     S^HSW = S^HT
            #     W = (S^HS)^-1 S^HT
            #  -> W = (S^HS + lamda I)^-1 S^HT
            # Notice that this W is a transposed version of the
            # above formulation.  Need to figure out if W @ S or
            # S @ W is more efficient matrix multiplication.
            # Currently computing W @ S when applying weights.
            S = A[:, P[ii, ...]]
            T = A[:, kx2, ky2, :]
            ShS = S.conj().T @ S
            ShT = S.conj().T @ T
            lamda0 = lamda*np.linalg.norm(ShS)/ShS.shape[0]
            W = np.linalg.solve(
                ShS + lamda0*np.eye(ShS.shape[0]), ShT).T

            # Now that we know the weights, let's apply them!  Find
            # all holes corresponding to current geometry.
            # Currently we're looping through all the points
            # associated with the current geometry.  It would be nice
            # to find a way to apply the weights to everything at
            # once.  Right now I don't know how to simultaneously
            # pull all source patches from kspace faster than a
            # for loop...

            # x, y define where top left corner is, so move to ctr,
            # also make sure they are iterable by enforcing atleast_1d
            idx = np.unravel_index(
                np.argwhere(iidx == ii), Psh[:2])
            x, y = idx[0]+kx2, idx[1]+ky2
            x = np.atleast_1d(x.squeeze())
            y = np.atleast_1d(y.squeeze())
            for xx, yy in zip(x, y):
                # Collect sources for this hole and apply weights
                S = kspace[xx-kx2:xx+kx2+adjx, yy-ky2:yy+ky2+adjy, :]
                S = S[P[ii, ...]]
                recon[xx, yy, :] = (W @ S[:, None]).squeeze()

        # Report on how long it took to train and apply weights
        if not silent:
            print(('Training and application of weights took %g'
                   'seconds' % (time() - t0)))

        # The recon array has been zero padded, so let's crop it down
        # to size and return it either as a memmap to the correct
        # file or in memory.
        # Also fill in known data, crop, move coil axis back.
        if memmap:
            fin = np.memmap(
                memmap_filename, dtype=recon.dtype, mode='w+',
                shape=fin_shape)
            fin[:] = np.moveaxis(
                (recon + kspace)[kx2:-kx2, ky2:-ky2, :],
                -1, coil_axis)
            del fin
            return None

        return np.moveaxis(
            (recon[:] + kspace)[kx2:-kx2, ky2:-ky2, :], -1, coil_axis)

def apply_grappa(masked_kspace, mask):
    """
    Applies GRAPPA algorithm
    References
    ----------
    [1] Griswold, Mark A., et al. "Generalized autocalibrating
       partially parallel acquisitions (GRAPPA)." Magnetic
       Resonance in Medicine: An Official Journal of the
       International Society for Magnetic Resonance in Medicine
       47.6 (2002): 1202-1210.
    Args:
        masked_kspace (torch.Tensor): Multi-coil masked input k-space of shape (num_coils, rows, cols, 2)
        mask (torch.Tensor): Applied mask of shape (1, 1, cols, 1)
    Returns:
        preprocessed_masked_kspace (torch.Tensor): Output of GRAPPA algorithm applied on masked_kspace
    """

    def get_low_frequency_lines(mask):
        l = r = mask.shape[-2] // 2
        while mask[..., r, :]:
            r += 1

        while mask[..., l, :]:
            l -= 1

        return l + 1, r

    l, r = get_low_frequency_lines(mask)
    num_low_freqs = r - l
    pad = (mask.shape[-2] - num_low_freqs + 1) // 2
    calib = masked_kspace[:, :, pad:pad + num_low_freqs].clone()
    preprocessed_masked_kspace = grappa(tensor_to_complex_np(masked_kspace), tensor_to_complex_np(calib),
                                        kernel_size=(3, 2), coil_axis=0)
    return to_tensor(preprocessed_masked_kspace)

def to_tensor(data: np.ndarray) -> torch.Tensor:
    """
    Convert numpy array to PyTorch tensor.

    For complex arrays, the real and imaginary parts are stacked along the last
    dimension.

    Args:
        data: Input numpy array.

    Returns:
        PyTorch version of data.
    """
    if np.iscomplexobj(data):
        data = np.stack((data.real, data.imag), axis=-1)

    return torch.from_numpy(data)


def tensor_to_complex_np(data: torch.Tensor) -> np.ndarray:
    """
    Converts a complex torch tensor to numpy array.

    Args:
        data: Input data to be converted to numpy.

    Returns:
        Complex numpy version of data.
    """
    data = data.numpy()

    return data[..., 0] + 1j * data[..., 1]



def apply_mask(
    data: torch.Tensor,
    mask_func: MaskFunc,
    offset: Optional[int] = None,
    seed: Optional[Union[int, Tuple[int, ...]]] = None,
    padding: Optional[Sequence[int]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Subsample given k-space by multiplying with a mask.
    Args:
        data: The input k-space data. This should have at least 3 dimensions,
            where dimensions -3 and -2 are the spatial dimensions, and the
            final dimension has size 2 (for complex values).
        mask_func: A function that takes a shape (tuple of ints) and a random
            number seed and returns a mask.
        seed: Seed for the random number generator.
        padding: Padding value to apply for mask.
    Returns:
        tuple containing:
            masked data: Subsampled k-space data.
            mask: The generated mask.
            num_low_frequencies: The number of low-resolution frequency samples
                in the mask.
    """
    shape = (1,) * len(data.shape[:-3]) + tuple(data.shape[-3:])
    mask, num_low_frequencies = mask_func(shape, offset, seed)
    if padding is not None:
        mask[..., : padding[0], :] = 0
        mask[..., padding[1] :, :] = 0  # padding value inclusive on right of zeros

    masked_data = data * mask + 0.0  # the + 0.0 removes the sign of the zeros

    return masked_data, mask, num_low_frequencies


def mask_center(x: torch.Tensor, mask_from: int, mask_to: int) -> torch.Tensor:
    """
    Initializes a mask with the center filled in.

    Args:
        mask_from: Part of center to start filling.
        mask_to: Part of center to end filling.

    Returns:
        A mask with the center filled.
    """
    mask = torch.zeros_like(x)
    mask[:,  mask_from:mask_to,:] = x[:,  mask_from:mask_to,:]

    return mask


def center_crop(data: torch.Tensor, shape: Tuple[int, int]) -> torch.Tensor:
    """
    Apply a center crop to the input real image or batch of real images.

    Args:
        data: The input tensor to be center cropped. It should
            have at least 2 dimensions and the cropping is applied along the
            last two dimensions.
        shape: The output shape. The shape should be smaller
            than the corresponding dimensions of data.

    Returns:
        The center cropped image.
    """
    if not (0 < shape[0] <= data.shape[-2] and 0 < shape[1] <= data.shape[-1]):
        raise ValueError("Invalid shapes.")

    w_from = (data.shape[-2] - shape[0]) // 2
    h_from = (data.shape[-1] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]

    return data[..., w_from:w_to, h_from:h_to]


def complex_center_crop(data: torch.Tensor, shape: Tuple[int, int]) -> torch.Tensor:
    """
    Apply a center crop to the input image or batch of complex images.

    Args:
        data: The complex input tensor to be center cropped. It should have at
            least 3 dimensions and the cropping is applied along dimensions -3
            and -2 and the last dimensions should have a size of 2.
        shape: The output shape. The shape should be smaller than the
            corresponding dimensions of data.

    Returns:
        The center cropped image
    """
    if not (0 < shape[0] <= data.shape[-3] and 0 < shape[1] <= data.shape[-2]):
        raise ValueError("Invalid shapes.")

    w_from = (data.shape[-3] - shape[0]) // 2
    h_from = (data.shape[-2] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]

    return data[..., w_from:w_to, h_from:h_to, :]


def i_complex_center_crop(data: torch.Tensor, shape: Tuple[int, int], data_croped: torch.Tensor) -> torch.Tensor:
    """
    Apply a center crop to the input image or batch of complex images.

    Args:
        data: The complex input tensor to be center cropped. It should have at
            least 3 dimensions and the cropping is applied along dimensions -3
            and -2 and the last dimensions should have a size of 2.
        shape: The output shape. The shape should be smaller than the
            corresponding dimensions of data.

    Returns:
        The center cropped image
    """

    w_from = (data.shape[-2] - shape[0]) // 2
    h_from = (data.shape[-1] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]
    data[..., :, w_from:w_to, h_from:h_to] = data_croped[...]

    return data


def center_crop_to_smallest(
        x: torch.Tensor, y: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply a center crop on the larger image to the size of the smaller.

    The minimum is taken over dim=-1 and dim=-2. If x is smaller than y at
    dim=-1 and y is smaller than x at dim=-2, then the returned dimension will
    be a mixture of the two.

    Args:
        x: The first image.
        y: The second image.

    Returns:
        tuple of tensors x and y, each cropped to the minimim size.
    """
    smallest_width = min(x.shape[-1], y.shape[-1])
    smallest_height = min(x.shape[-2], y.shape[-2])
    x = center_crop(x, (smallest_height, smallest_width))
    y = center_crop(y, (smallest_height, smallest_width))

    return x, y


def normalize(
        data: torch.Tensor,
        mean: Union[float, torch.Tensor],
        stddev: Union[float, torch.Tensor],
        eps: Union[float, torch.Tensor] = 0.0,
) -> torch.Tensor:
    """
    Normalize the given tensor.

    Applies the formula (data - mean) / (stddev + eps).

    Args:
        data: Input data to be normalized.
        mean: Mean value.
        stddev: Standard deviation.
        eps: Added to stddev to prevent dividing by zero.

    Returns:
        Normalized tensor.
    """
    return (data - mean) / (stddev + eps)


def normalize_instance(
        data: torch.Tensor, eps: Union[float, torch.Tensor] = 0.0
) -> Tuple[torch.Tensor, Union[torch.Tensor], Union[torch.Tensor]]:
    """
    Normalize the given tensor  with instance norm/

    Applies the formula (data - mean) / (stddev + eps), where mean and stddev
    are computed from the data itself.

    Args:
        data: Input data to be normalized
        eps: Added to stddev to prevent dividing by zero.

    Returns:
        torch.Tensor: Normalized tensor
    """
    mean = data.mean()
    std = data.std()

    return normalize(data, mean, std, eps), mean, std


class UnetDataTargetTransform:
    """
    Data Transformer for training U-Net models.
    """

    def __init__(
            self,
            which_challenge: str,
            mask_func: Optional[MaskFunc] = None,
            use_seed: bool = True,
    ):
        """
        Args:
            which_challenge: Challenge from ("singlecoil", "multicoil").
            mask_func: Optional; A function that can create a mask of
                appropriate shape.
            use_seed: If true, this class computes a pseudo random number
                generator seed from the filename. This ensures that the same
                mask is used for all the slices of a given volume every time.
        """
        if which_challenge not in ("singlecoil", "multicoil"):
            raise ValueError("Challenge should either be 'singlecoil' or 'multicoil'")

        self.mask_func = mask_func
        self.which_challenge = which_challenge
        self.use_seed = use_seed

    def __call__(
            self,
            kspace: np.ndarray,
            mask: np.ndarray,
            target: np.ndarray,
            attrs: Dict,
            fname: str,
            slice_num: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str, int, float]:
        """
        Args:
            kspace: Input k-space of shape (num_coils, rows, cols) for
                multi-coil data or (rows, cols) for single coil data.
            mask: Mask from the test dataset.
            target: Target image.
            attrs: Acquisition related information stored in the HDF5 object.
            fname: File name.
            slice_num: Serial number of the slice.

        Returns:
            tuple containing:
                image: Zero-filled input image.
                target: Target image converted to a torch.Tensor.
                mean: Mean value used for normalization.
                std: Standard deviation value used for normalization.
                fname: File name.
                slice_num: Serial number of the slice.
        """
        import h5py
        hf = h5py.File('/home/biit/fastmri_dataset/singlecoil_knee1/singlecoil_val/file1000277.h5')
        target = hf['reconstruction_rss'][()][11]

        kspace = to_tensor(kspace)

        # check for max value
        max_value = attrs["max"] if "max" in attrs.keys() else 0.0

        # apply mask
        if self.mask_func:
            seed = None if not self.use_seed else tuple(map(ord, fname))
            masked_kspace, mask = apply_mask(kspace, self.mask_func, seed)
        else:
            masked_kspace = kspace

        # inverse Fourier transform to get zero filled solution
        image = fastmri.ifft2c(masked_kspace)
        # image1 = fastmri.complex_abs(image)
        # from matplotlib import pyplot as plt
        # plt.imshow(image1,'gray')
        # plt.savefig('./test.png')
        # plt.imshow(target,'gray')
        # plt.savefig('./test1.png')

        # crop input to correct size
        # if target is not None:
        #     crop_size = (target.shape[-2], target.shape[-1])
        # else:
        #     crop_size = (attrs["recon_size"][0], attrs["recon_size"][1])

        # check for FLAIR 203
        # if image.shape[-2] < crop_size[1]:
        #     crop_size = (image.shape[-2], image.shape[-2])

        # image = complex_center_crop(image, crop_size)

        # absolute value
        # image = fastmri.complex_abs(image)

        # apply Root-Sum-of-Squares if multicoil data
        # if self.which_challenge == "multicoil":
        #     image = fastmri.rss(image)

        # normalize input
        image = image.permute(2, 0, 1)
        # image, mean, std = self.norm(image)

        # normalize target
        if target is not None:
            target = to_tensor(target)
            # target = center_crop(target, crop_size)
            target, mean, std = normalize_instance(target, eps=1e-11)
            # target, mean_target, std_target = normalize_instance(target, eps=1e-11)
            # target = target.clamp(-6, 6)
        else:
            target = torch.Tensor([0])

        return target, target, mean, std, fname, slice_num, max_value


class UnetDataTransform:
    """
    Data Transformer for training U-Net models.
    """

    def __init__(
            self,
            which_challenge: str,
            mask_func: Optional[MaskFunc] = None,
            use_seed: bool = True,
    ):
        """
        Args:
            which_challenge: Challenge from ("singlecoil", "multicoil").
            mask_func: Optional; A function that can create a mask of
                appropriate shape.
            use_seed: If true, this class computes a pseudo random number
                generator seed from the filename. This ensures that the same
                mask is used for all the slices of a given volume every time.
        """
        if which_challenge not in ("singlecoil", "multicoil"):
            raise ValueError("Challenge should either be 'singlecoil' or 'multicoil'")

        self.mask_func = mask_func
        self.which_challenge = which_challenge
        self.use_seed = use_seed

    def __call__(
            self,
            kspace: np.ndarray,
            mask: np.ndarray,
            target: np.ndarray,
            attrs: Dict,
            fname: str,
            slice_num: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str, int, float]:
        """
        Args:
            kspace: Input k-space of shape (num_coils, rows, cols) for
                multi-coil data or (rows, cols) for single coil data.
            mask: Mask from the test dataset.
            target: Target image.
            attrs: Acquisition related information stored in the HDF5 object.
            fname: File name.
            slice_num: Serial number of the slice.

        Returns:
            tuple containing:
                image: Zero-filled input image.
                target: Target image converted to a torch.Tensor.
                mean: Mean value used for normalization.
                std: Standard deviation value used for normalization.
                fname: File name.
                slice_num: Serial number of the slice.
        """
        kspace = to_tensor(kspace)

        # check for max value
        max_value = attrs["max"] if "max" in attrs.keys() else 0.0

        # apply mask
        if self.mask_func:
            seed = None if not self.use_seed else tuple(map(ord, fname))
            masked_kspace, mask = apply_mask(kspace, self.mask_func, seed)
        else:
            masked_kspace = kspace
            mask = torch.from_numpy(mask).unsqueeze(1).unsqueeze(0)

        # inverse Fourier transform to get zero filled solution
        # image = fastmri.ifft2c(masked_kspace)

        # crop input to correct size
        # if target is not None:
        #     crop_size = (target.shape[-2], target.shape[-1])
        # else:
        #     crop_size = (attrs["recon_size"][0], attrs["recon_size"][1])

        # check for FLAIR 203
        # if image.shape[-2] < crop_size[1]:
        #     crop_size = (image.shape[-2], image.shape[-2])

        # image = complex_center_crop(image, crop_size)

        # absolute value
        # image = fastmri.complex_abs(image)

        # apply Root-Sum-of-Squares if multicoil data
        # if self.which_challenge == "multicoil":
        #     image = fastmri.rss(image)

        # normalize input
        image = masked_kspace.permute(2, 0, 1)
        # image, mean, std = self.norm(image)

        # normalize target
        if target is not None:
            target = to_tensor(target)
            # target = center_crop(target, crop_size)
            # target = normalize(target, mean, std, eps=1e-11)
            # target, mean_target, std_target = normalize_instance(target, eps=1e-11)
            # target = target.clamp(-6, 6)
        else:
            target = torch.Tensor([0])

        return image, target, 0, 0, fname, slice_num, max_value, mask.permute(2, 0, 1).byte(), kspace.permute(2, 0, 1)


class UnetDataTransform_MC:
    """
    Data Transformer for training U-Net models.
    """

    def __init__(
            self,
            which_challenge: str,
            mask_func: Optional[MaskFunc] = None,
            use_seed: bool = True,
    ):
        """
        Args:
            which_challenge: Challenge from ("singlecoil", "multicoil").
            mask_func: Optional; A function that can create a mask of
                appropriate shape.
            use_seed: If true, this class computes a pseudo random number
                generator seed from the filename. This ensures that the same
                mask is used for all the slices of a given volume every time.
        """
        if which_challenge not in ("singlecoil", "multicoil"):
            raise ValueError("Challenge should either be 'singlecoil' or 'multicoil'")

        self.mask_func = mask_func
        self.which_challenge = which_challenge
        self.use_seed = use_seed

    def __call__(
            self,
            kspace: np.ndarray,
            mask: np.ndarray,
            target: np.ndarray,
            attrs: Dict,
            fname: str,
            slice_num: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str, int, float]:
        """
        Args:
            kspace: Input k-space of shape (num_coils, rows, cols) for
                multi-coil data or (rows, cols) for single coil data.
            mask: Mask from the test dataset.
            target: Target image.
            attrs: Acquisition related information stored in the HDF5 object.
            fname: File name.
            slice_num: Serial number of the slice.

        Returns:
            tuple containing:
                image: Zero-filled input image.
                target: Target image converted to a torch.Tensor.
                mean: Mean value used for normalization.
                std: Standard deviation value used for normalization.
                fname: File name.
                slice_num: Serial number of the slice.
        """
        kspace = to_tensor(kspace)

        # check for max value
        max_value = attrs["max"] if "max" in attrs.keys() else 0.0

        # apply mask
        if self.mask_func:
            seed = None if not self.use_seed else tuple(map(ord, fname))
            masked_kspace, mask = apply_mask(kspace, self.mask_func, seed)
        else:
            masked_kspace = kspace
            mask = torch.from_numpy(mask).unsqueeze(1).unsqueeze(0)

        # inverse Fourier transform to get zero filled solution
        # image = fastmri.ifft2c(masked_kspace)

        # crop input to correct size
        # if target is not None:
        #     crop_size = (target.shape[-2], target.shape[-1])
        # else:
        #     crop_size = (attrs["recon_size"][0], attrs["recon_size"][1])

        # check for FLAIR 203
        # if image.shape[-2] < crop_size[1]:
        #     crop_size = (image.shape[-2], image.shape[-2])

        # image = complex_center_crop(image, crop_size)

        # absolute value
        # image = fastmri.complex_abs(image)

        # apply Root-Sum-of-Squares if multicoil data
        # if self.which_challenge == "multicoil":
        #     image = fastmri.rss(image)

        # normalize input
        # image = masked_kspace.permute(2,0,1)
        # image, mean, std = self.norm(image)

        # normalize target
        if target is not None:
            target = to_tensor(target)
            # target = center_crop(target, crop_size)
            # target = normalize(target, mean, std, eps=1e-11)
            # target, mean_target, std_target = normalize_instance(target, eps=1e-11)
            # target = target.clamp(-6, 6)
        else:
            target = torch.Tensor([0])

        return masked_kspace, target, 0, 0, fname, slice_num, max_value, mask.byte()


class UnetDataRawTransform:
    """
    Data Transformer for training U-Net models.
    """

    def __init__(
            self,
            which_challenge: str,
            mask_func: Optional[MaskFunc] = None,
            use_seed: bool = True,
    ):
        """
        Args:
            which_challenge: Challenge from ("singlecoil", "multicoil").
            mask_func: Optional; A function that can create a mask of
                appropriate shape.
            use_seed: If true, this class computes a pseudo random number
                generator seed from the filename. This ensures that the same
                mask is used for all the slices of a given volume every time.
        """
        if which_challenge not in ("singlecoil", "multicoil"):
            raise ValueError("Challenge should either be 'singlecoil' or 'multicoil'")

        self.mask_func = mask_func
        self.which_challenge = which_challenge
        self.use_seed = use_seed

    def __call__(
            self,
            kspace: np.ndarray,
            mask: np.ndarray,
            target: np.ndarray,
            attrs: Dict,
            fname: str,
            slice_num: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str, int, float]:
        """
        Args:
            kspace: Input k-space of shape (num_coils, rows, cols) for
                multi-coil data or (rows, cols) for single coil data.
            mask: Mask from the test dataset.
            target: Target image.
            attrs: Acquisition related information stored in the HDF5 object.
            fname: File name.
            slice_num: Serial number of the slice.

        Returns:
            tuple containing:
                image: Zero-filled input image.
                target: Target image converted to a torch.Tensor.
                mean: Mean value used for normalization.
                std: Standard deviation value used for normalization.
                fname: File name.
                slice_num: Serial number of the slice.
        """
        kspace = to_tensor(kspace)

        # check for max value
        max_value = attrs["max"] if "max" in attrs.keys() else 0.0

        # apply mask
        if self.mask_func:
            seed = None if not self.use_seed else tuple(map(ord, fname))
            masked_kspace, mask = apply_mask(kspace, self.mask_func, seed)
        else:
            masked_kspace = kspace
            mask = torch.from_numpy(mask).unsqueeze(1).unsqueeze(0)

        # inverse Fourier transform to get zero filled solution
        image = fastmri.ifft2c(masked_kspace)

        # crop input to correct size
        # if target is not None:
        #     crop_size = (target.shape[-2], target.shape[-1])
        # else:
        #     crop_size = (attrs["recon_size"][0], attrs["recon_size"][1])

        # check for FLAIR 203
        # if image.shape[-2] < crop_size[1]:
        #     crop_size = (image.shape[-2], image.shape[-2])

        # image = complex_center_crop(image, crop_size)

        # absolute value
        # image = fastmri.complex_abs(image)

        # apply Root-Sum-of-Squares if multicoil data
        # if self.which_challenge == "multicoil":
        #     image = fastmri.rss(image)

        # normalize input
        image = image.permute(2, 0, 1)
        # image, mean, std = self.norm(image)

        # normalize target
        if target is not None:
            target = to_tensor(target)
            # target = center_crop(target, crop_size)
            # target = normalize(target, mean, std, eps=1e-11)
            # target, mean_target, std_target = normalize_instance(target, eps=1e-11)
            # target = target.clamp(-6, 6)
        else:
            target = torch.Tensor([0])

        return image, target, 0, 0, fname, slice_num, max_value, mask.permute(2, 0, 1).byte()


class XiaoboTargetDataTransform:
    """
    Data Transformer for training U-Net models.
    """

    def __init__(
            self,
            which_challenge: str,
            mask_func: Optional[MaskFunc] = None,
            use_seed: bool = True,
    ):
        """
        Args:
            which_challenge: Challenge from ("singlecoil", "multicoil").
            mask_func: Optional; A function that can create a mask of
                appropriate shape.
            use_seed: If true, this class computes a pseudo random number
                generator seed from the filename. This ensures that the same
                mask is used for all the slices of a given volume every time.
        """
        if which_challenge not in ("singlecoil", "multicoil"):
            raise ValueError("Challenge should either be 'singlecoil' or 'multicoil'")

        self.mask_func = mask_func
        self.which_challenge = which_challenge
        self.use_seed = use_seed

        self.xfm = DWTForward(J=1, mode='zero', wave='haar')
        self.ifm = DWTInverse(mode='zero', wave='haar')

    def __call__(
            self,
            kspace: np.ndarray,
            mask: np.ndarray,
            target: np.ndarray,
            attrs: Dict,
            fname: str,
            slice_num: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str, int, float]:
        """
        Args:
            kspace: Input k-space of shape (num_coils, rows, cols) for
                multi-coil data or (rows, cols) for single coil data.
            mask: Mask from the test dataset.
            target: Target image.
            attrs: Acquisition related information stored in the HDF5 object.
            fname: File name.
            slice_num: Serial number of the slice.

        Returns:
            tuple containing:
                image: Zero-filled input image.
                target: Target image converted to a torch.Tensor.
                mean: Mean value used for normalization.
                std: Standard deviation value used for normalization.
                fname: File name.
                slice_num: Serial number of the slice.
        """
        kspace = to_tensor(kspace)

        # check for max value
        max_value = attrs["max"] if "max" in attrs.keys() else 0.0

        # apply mask
        if self.mask_func:
            seed = None if not self.use_seed else tuple(map(ord, fname))
            masked_kspace, mask = apply_mask(kspace, self.mask_func, seed)
        else:
            masked_kspace = kspace

        # inverse Fourier transform to get zero filled solution
        image = fastmri.ifft2c(masked_kspace)

        # normalize input
        image = image.permute(2, 0, 1)

        # normalize target
        if target is not None:
            target = to_tensor(target)
            ll, hs = self.xfm(target.unsqueeze(0).unsqueeze(0))
            hs = hs[0][0]
            haar_image = torch.cat([ll, hs], 1)[0]
            # print(0)
            # target = center_crop(target, crop_size)
            # target = normalize(target, mean, std, eps=1e-11)
            # target, mean_target, std_target = normalize_instance(target, eps=1e-11)
            # target = target.clamp(-6, 6)
        else:
            target = torch.Tensor([0])

        return image, target, 0, 0, fname, slice_num, max_value, mask.permute(2, 0, 1).byte(), haar_image


class KspaceDataTransform:
    """
    Data Transformer for training U-Net models.
    """

    def __init__(
            self,
            which_challenge: str,
            mask_func: Optional[MaskFunc] = None,
            use_seed: bool = True,
    ):
        """
        Args:
            which_challenge: Challenge from ("singlecoil", "multicoil").
            mask_func: Optional; A function that can create a mask of
                appropriate shape.
            use_seed: If true, this class computes a pseudo random number
                generator seed from the filename. This ensures that the same
                mask is used for all the slices of a given volume every time.
        """
        if which_challenge not in ("singlecoil", "multicoil"):
            raise ValueError("Challenge should either be 'singlecoil' or 'multicoil'")

        self.mask_func = mask_func
        self.which_challenge = which_challenge
        self.use_seed = use_seed

    def __call__(
            self,
            kspace: np.ndarray,
            mask: np.ndarray,
            target: np.ndarray,
            attrs: Dict,
            fname: str,
            slice_num: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str, int, float]:
        """
        Args:
            kspace: Input k-space of shape (num_coils, rows, cols) for
                multi-coil data or (rows, cols) for single coil data.
            mask: Mask from the test dataset.
            target: Target image.
            attrs: Acquisition related information stored in the HDF5 object.
            fname: File name.
            slice_num: Serial number of the slice.

        Returns:
            tuple containing:
                image: Zero-filled input image.
                target: Target image converted to a torch.Tensor.
                mean: Mean value used for normalization.
                std: Standard deviation value used for normalization.
                fname: File name.
                slice_num: Serial number of the slice.
        """
        kspace = to_tensor(kspace)

        # check for max value
        max_value = attrs["max"] if "max" in attrs.keys() else 0.0

        # apply mask
        if self.mask_func:
            seed = None if not self.use_seed else tuple(map(ord, fname))
            masked_kspace, mask = apply_mask(kspace, self.mask_func, seed)
        else:
            masked_kspace = kspace

        # inverse Fourier transform to get zero filled solution
        # image = fastmri.ifft2c(masked_kspace)

        # crop input to correct size
        # if target is not None:
        #     crop_size = (target.shape[-2], target.shape[-1])
        # else:
        #     crop_size = (attrs["recon_size"][0], attrs["recon_size"][1])

        # check for FLAIR 203
        # if image.shape[-2] < crop_size[1]:
        #     crop_size = (image.shape[-2], image.shape[-2])

        # image = complex_center_crop(image, crop_size)

        # absolute value
        # image = fastmri.complex_abs(image)

        # apply Root-Sum-of-Squares if multicoil data
        # if self.which_challenge == "multicoil":
        #     image = fastmri.rss(image)

        # normalize input
        masked_kspace = masked_kspace.permute(2, 0, 1)
        # image, mean, std = self.norm(image)

        # normalize target
        if target is not None:
            target = to_tensor(target)
            # target = center_crop(target, crop_size)
            # target = normalize(target, mean, std, eps=1e-11)
            # target, mean_target, std_target = normalize_instance(target, eps=1e-11)
            # target = target.clamp(-6, 6)
        else:
            target = torch.Tensor([0])

        return masked_kspace, target, 0, 0, fname, slice_num, max_value


class FcData32Transform:
    """
    Data Transformer for training U-Net models.
    """

    def __init__(
            self,
            which_challenge: str,
            mask_func: Optional[MaskFunc] = None,
            use_seed: bool = True,
    ):
        """
        Args:
            which_challenge: Challenge from ("singlecoil", "multicoil").
            mask_func: Optional; A function that can create a mask of
                appropriate shape.
            use_seed: If true, this class computes a pseudo random number
                generator seed from the filename. This ensures that the same
                mask is used for all the slices of a given volume every time.
        """
        if which_challenge not in ("singlecoil", "multicoil"):
            raise ValueError("Challenge should either be 'singlecoil' or 'multicoil'")

        self.mask_func = mask_func
        self.which_challenge = which_challenge
        self.use_seed = use_seed

    def __call__(
            self,
            kspace: np.ndarray,
            mask: np.ndarray,
            target: np.ndarray,
            attrs: Dict,
            fname: str,
            slice_num: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str, int, float]:
        """
        Args:
            kspace: Input k-space of shape (num_coils, rows, cols) for
                multi-coil data or (rows, cols) for single coil data.
            mask: Mask from the test dataset.
            target: Target image.
            attrs: Acquisition related information stored in the HDF5 object.
            fname: File name.
            slice_num: Serial number of the slice.

        Returns:
            tuple containing:
                image: Zero-filled input image.
                target: Target image converted to a torch.Tensor.
                mean: Mean value used for normalization.
                std: Standard deviation value used for normalization.
                fname: File name.
                slice_num: Serial number of the slice.
        """
        kspace = to_tensor(kspace)

        # check for max value
        max_value = attrs["max"] if "max" in attrs.keys() else 0.0

        # apply mask
        if self.mask_func:
            seed = None if not self.use_seed else tuple(map(ord, fname))
            masked_kspace, mask = apply_mask(kspace, self.mask_func, seed)
        else:
            masked_kspace = kspace

        # inverse Fourier transform to get zero filled solution
        image = fastmri.ifft2c(masked_kspace)
        image = center_crop(image.permute(2, 0, 1), [320, 320])
        image_target = fastmri.ifft2c(kspace)
        image_target = center_crop(image_target.permute(2, 0, 1), [320, 320])

        # crop input to correct size
        # if target is not None:
        #     crop_size = (target.shape[-2], target.shape[-1])
        # else:
        #     crop_size = (attrs["recon_size"][0], attrs["recon_size"][1])

        # check for FLAIR 203
        # if image.shape[-2] < crop_size[1]:
        #     crop_size = (image.shape[-2], image.shape[-2])

        # image = complex_center_crop(image, crop_size)

        # absolute value
        # image = fastmri.complex_abs(image)

        # apply Root-Sum-of-Squares if multicoil data
        # if self.which_challenge == "multicoil":
        #     image = fastmri.rss(image)

        # normalize input
        # image = image.permute(2,0,1)
        # image_target = image_target.permute(2,0,1)
        image = F.interpolate(image.unsqueeze(0), (32, 32), mode='bilinear').squeeze(0)
        # image = fastmri.fft2c(image.permute(1,2,0)).permute(2,0,1)
        image_target = F.interpolate(image_target.unsqueeze(0), (32, 32), mode='bilinear').squeeze(0)
        image_target = fastmri.fft2c(image_target.permute(1, 2, 0)).permute(2, 0, 1)
        # image = fastmri.complex_abs(image.permute(0,2,3,1))
        # from matplotlib import pyplot as plt
        # plt.imshow(image[0], 'gray')
        # plt.savefig('./test.png')
        # image, mean, std = self.norm(image)

        # normalize target
        if target is not None:
            target = to_tensor(target)
            target = F.interpolate(target.unsqueeze(0).unsqueeze(0), (32, 32), mode='bilinear').squeeze(0)
            # from matplotlib import pyplot as plt
            # plt.imshow(target[0], 'gray')
            # plt.savefig('./test.png')
            # target = center_crop(target, crop_size)
            # target = normalize(target, mean, std, eps=1e-11)
            # target, mean_target, std_target = normalize_instance(target, eps=1e-11)
            # target = target.clamp(-6, 6)
        else:
            target = torch.Tensor([0])

        return target, target, 0, 0, fname, slice_num, max_value, image_target


def guideFilter(I, p, winSize, eps, s):
    # 输入图像的高、宽
    h, w = I.shape[:2]

    # 缩小图像
    size = (int(round(w * s)), int(round(h * s)))

    small_I = cv2.resize(I, size, interpolation=cv2.INTER_CUBIC)
    small_p = cv2.resize(I, size, interpolation=cv2.INTER_CUBIC)

    # 缩小滑动窗口
    X = winSize[0]
    small_winSize = (int(round(X * s)), int(round(X * s)))

    # I的均值平滑
    mean_small_I = cv2.blur(small_I, small_winSize)

    # p的均值平滑
    mean_small_p = cv2.blur(small_p, small_winSize)

    # I*I和I*p的均值平滑
    mean_small_II = cv2.blur(small_I * small_I, small_winSize)

    mean_small_Ip = cv2.blur(small_I * small_p, small_winSize)

    # 方差
    var_small_I = mean_small_II - mean_small_I * mean_small_I  # 方差公式

    # 协方差
    cov_small_Ip = mean_small_Ip - mean_small_I * mean_small_p

    small_a = cov_small_Ip / (var_small_I + eps)
    small_b = mean_small_p - small_a * mean_small_I

    # 对a、b进行均值平滑
    mean_small_a = cv2.blur(small_a, small_winSize)
    mean_small_b = cv2.blur(small_b, small_winSize)

    # 放大
    size1 = (w, h)
    mean_a = cv2.resize(mean_small_a, size1, interpolation=cv2.INTER_LINEAR)
    mean_b = cv2.resize(mean_small_b, size1, interpolation=cv2.INTER_LINEAR)

    q = mean_a * I + mean_b

    return q


class UnetDataWithLowFreqTargetTransform:
    """
    Data Transformer for training U-Net models.
    """

    def __init__(
            self,
            which_challenge: str,
            mask_func: Optional[MaskFunc] = None,
            use_seed: bool = True,
    ):
        """
        Args:
            which_challenge: Challenge from ("singlecoil", "multicoil").
            mask_func: Optional; A function that can create a mask of
                appropriate shape.
            use_seed: If true, this class computes a pseudo random number
                generator seed from the filename. This ensures that the same
                mask is used for all the slices of a given volume every time.
        """
        if which_challenge not in ("singlecoil", "multicoil"):
            raise ValueError("Challenge should either be 'singlecoil' or 'multicoil'")

        self.mask_func = mask_func
        self.which_challenge = which_challenge
        self.use_seed = use_seed

    def __call__(
            self,
            kspace: np.ndarray,
            mask: np.ndarray,
            target: np.ndarray,
            attrs: Dict,
            fname: str,
            slice_num: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str, int, float]:
        """
        Args:
            kspace: Input k-space of shape (num_coils, rows, cols) for
                multi-coil data or (rows, cols) for single coil data.
            mask: Mask from the test dataset.
            target: Target image.
            attrs: Acquisition related information stored in the HDF5 object.
            fname: File name.
            slice_num: Serial number of the slice.

        Returns:
            tuple containing:
                image: Zero-filled input image.
                target: Target image converted to a torch.Tensor.
                mean: Mean value used for normalization.
                std: Standard deviation value used for normalization.
                fname: File name.
                slice_num: Serial number of the slice.
        """
        kspace = to_tensor(kspace)

        # check for max value
        max_value = attrs["max"] if "max" in attrs.keys() else 0.0

        # apply mask
        if self.mask_func:
            seed = None if not self.use_seed else tuple(map(ord, fname))
            masked_kspace, mask = apply_mask(kspace, self.mask_func, seed)
        else:
            masked_kspace = kspace

        # inverse Fourier transform to get zero filled solution
        image = fastmri.ifft2c(masked_kspace).permute(2, 0, 1)

        # normalize input
        # image = fastmri.complex_abs(image).unsqueeze(0)
        # image, mean, std = normalize_instance(image, eps=1e-11)
        # image = image.clamp(-6, 6)

        # normalize target
        if target is not None:
            target = to_tensor(target)
            target_max = torch.max(target)
            target_min = torch.min(target)
            target_guiyi = ((target - target_min) / target_max).numpy()
            target_guiyi = torch.from_numpy(guideFilter(target_guiyi, target_guiyi, (16, 16), 0.01, 0.5)).unsqueeze(
                0) * target_max + target_min
            # from matplotlib import pyplot as plt
            # plt.imshow(target_guiyi, 'gray')
            # plt.savefig('./test.png')
            # target = normalize(target.unsqueeze(0), mean, std, eps=1e-11)
            # target = target.clamp(-6, 6)
            # target_guiyi = normalize(target_guiyi, mean, std, eps=1e-11)
            # target_guiyi = target_guiyi.clamp(-6, 6)
        else:
            target = torch.Tensor([0])

        return image, target.unsqueeze(0), 0, 0, fname, slice_num, max_value, target_guiyi, mask


class UnetAbsDataTransform:
    """
    Data Transformer for training U-Net models.
    """

    def __init__(
            self,
            which_challenge: str,
            mask_func: Optional[MaskFunc] = None,
            use_seed: bool = True,
    ):
        """
        Args:
            which_challenge: Challenge from ("singlecoil", "multicoil").
            mask_func: Optional; A function that can create a mask of
                appropriate shape.
            use_seed: If true, this class computes a pseudo random number
                generator seed from the filename. This ensures that the same
                mask is used for all the slices of a given volume every time.
        """
        if which_challenge not in ("singlecoil", "multicoil"):
            raise ValueError("Challenge should either be 'singlecoil' or 'multicoil'")

        self.mask_func = mask_func
        self.which_challenge = which_challenge
        self.use_seed = use_seed

    def __call__(
            self,
            kspace: np.ndarray,
            mask: np.ndarray,
            target: np.ndarray,
            attrs: Dict,
            fname: str,
            slice_num: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str, int, float]:
        """
        Args:
            kspace: Input k-space of shape (num_coils, rows, cols) for
                multi-coil data or (rows, cols) for single coil data.
            mask: Mask from the test dataset.
            target: Target image.
            attrs: Acquisition related information stored in the HDF5 object.
            fname: File name.
            slice_num: Serial number of the slice.

        Returns:
            tuple containing:
                image: Zero-filled input image.
                target: Target image converted to a torch.Tensor.
                mean: Mean value used for normalization.
                std: Standard deviation value used for normalization.
                fname: File name.
                slice_num: Serial number of the slice.
        """
        kspace = to_tensor(kspace)

        # check for max value
        max_value = attrs["max"] if "max" in attrs.keys() else 0.0

        # apply mask
        if self.mask_func:
            seed = None if not self.use_seed else tuple(map(ord, fname))
            masked_kspace, mask = apply_mask(kspace, self.mask_func, seed)
        else:
            masked_kspace = kspace

        # inverse Fourier transform to get zero filled solution
        image = fastmri.ifft2c(masked_kspace)

        # crop input to correct size
        # if target is not None:
        #     crop_size = (target.shape[-2], target.shape[-1])
        # else:
        #     crop_size = (attrs["recon_size"][0], attrs["recon_size"][1])

        # check for FLAIR 203
        # if image.shape[-2] < crop_size[1]:
        #     crop_size = (image.shape[-2], image.shape[-2])

        # image = complex_center_crop(image, crop_size)

        # absolute value
        # image = fastmri.complex_abs(image)

        # apply Root-Sum-of-Squares if multicoil data
        # if self.which_challenge == "multicoil":
        #     image = fastmri.rss(image)

        # normalize input
        image = fastmri.complex_abs(image).unsqueeze(0)
        image = center_crop(image, [320, 320])
        image, mean, std = normalize_instance(image, eps=1e-11)
        # image, mean, std = self.norm(image)

        # normalize target
        if target is not None:
            target = to_tensor(target).unsqueeze(0)
            # target = center_crop(target, crop_size)
            target = normalize(target, mean, std, eps=1e-11)
            # target, mean_target, std_target = normalize_instance(target, eps=1e-11)
            # target = target.clamp(-6, 6)
        else:
            target = torch.Tensor([0])

        return image, target, mean, std, fname, slice_num, max_value


class UnetDataTransform_320:
    """
    Data Transformer for training U-Net models.
    """

    def __init__(
            self,
            which_challenge: str,
            mask_func: Optional[MaskFunc] = None,
            use_seed: bool = True,
    ):
        """
        Args:
            which_challenge: Challenge from ("singlecoil", "multicoil").
            mask_func: Optional; A function that can create a mask of
                appropriate shape.
            use_seed: If true, this class computes a pseudo random number
                generator seed from the filename. This ensures that the same
                mask is used for all the slices of a given volume every time.
        """
        if which_challenge not in ("singlecoil", "multicoil"):
            raise ValueError("Challenge should either be 'singlecoil' or 'multicoil'")

        self.mask_func = mask_func
        self.which_challenge = which_challenge
        self.use_seed = use_seed

    def __call__(
            self,
            kspace: np.ndarray,
            mask: np.ndarray,
            target: np.ndarray,
            attrs: Dict,
            fname: str,
            slice_num: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str, int, float]:
        """
        Args:
            kspace: Input k-space of shape (num_coils, rows, cols) for
                multi-coil data or (rows, cols) for single coil data.
            mask: Mask from the test dataset.
            target: Target image.
            attrs: Acquisition related information stored in the HDF5 object.
            fname: File name.
            slice_num: Serial number of the slice.

        Returns:
            tuple containing:
                image: Zero-filled input image.
                target: Target image converted to a torch.Tensor.
                mean: Mean value used for normalization.
                std: Standard deviation value used for normalization.
                fname: File name.
                slice_num: Serial number of the slice.
        """
        kspace = to_tensor(kspace)

        # check for max value
        max_value = attrs["max"] if "max" in attrs.keys() else 0.0

        # apply mask
        if self.mask_func:
            seed = None if not self.use_seed else tuple(map(ord, fname))
            masked_kspace, mask = apply_mask(kspace, self.mask_func, seed)
        else:
            masked_kspace = kspace

        # inverse Fourier transform to get zero filled solution
        image = fastmri.ifft2c(masked_kspace)

        # crop input to correct size
        # if target is not None:
        #     crop_size = (target.shape[-2], target.shape[-1])
        # else:
        #     crop_size = (attrs["recon_size"][0], attrs["recon_size"][1])

        # check for FLAIR 203
        # if image.shape[-2] < crop_size[1]:
        #     crop_size = (image.shape[-2], image.shape[-2])

        # image = complex_center_crop(image, crop_size)

        # absolute value
        # image = fastmri.complex_abs(image)

        # apply Root-Sum-of-Squares if multicoil data
        # if self.which_challenge == "multicoil":
        #     image = fastmri.rss(image)

        # normalize input
        image = complex_center_crop(image, [320, 320])
        image = image.permute(2, 0, 1)
        # image, mean, std = self.norm(image)

        # normalize target
        if target is not None:
            target = to_tensor(target)
            # target = center_crop(target, crop_size)
            # target = normalize(target, mean, std, eps=1e-11)
            # target, mean_target, std_target = normalize_instance(target, eps=1e-11)
            # target = target.clamp(-6, 6)
        else:
            target = torch.Tensor([0])

        return target.unsqueeze(0), target.unsqueeze(0), 0, 0, fname, slice_num, max_value, mask


class VarNetDataTransform:
    """
    Data Transformer for training VarNet models.
    """

    def __init__(self, mask_func: Optional[MaskFunc] = None, use_seed: bool = True):
        """
        Args:
            mask_func: Optional; A function that can create a mask of
                appropriate shape. Defaults to None.
            use_seed: If True, this class computes a pseudo random number
                generator seed from the filename. This ensures that the same
                mask is used for all the slices of a given volume every time.
        """
        self.mask_func = mask_func
        self.use_seed = use_seed

    def __call__(
            self,
            kspace: np.ndarray,
            mask: np.ndarray,
            target: np.ndarray,
            attrs: Dict,
            fname: str,
            slice_num: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str, int, float, torch.Tensor]:
        """
        Args:
            kspace: Input k-space of shape (num_coils, rows, cols) for
                multi-coil data.
            mask: Mask from the test dataset.
            target: Target image.
            attrs: Acquisition related information stored in the HDF5 object.
            fname: File name.
            slice_num: Serial number of the slice.

        Returns:
            tuple containing:
                masked_kspace: k-space after applying sampling mask.
                mask: The applied sampling mask
                target: The target image (if applicable).
                fname: File name.
                slice_num: The slice index.
                max_value: Maximum image value.
                crop_size: The size to crop the final image.
        """
        if target is not None:
            target = to_tensor(target)
            max_value = attrs["max"]
        else:
            target = torch.tensor(0)
            max_value = 0.0

        kspace = to_tensor(kspace)
        seed = None if not self.use_seed else tuple(map(ord, fname))
        acq_start = attrs["padding_left"]
        acq_end = attrs["padding_right"]

        crop_size = torch.tensor([attrs["recon_size"][0], attrs["recon_size"][1]])

        if self.mask_func:
            masked_kspace, mask = apply_mask(
                kspace, self.mask_func, seed, (acq_start, acq_end)
            )
        else:
            masked_kspace = kspace
            shape = np.array(kspace.shape)
            num_cols = shape[-2]
            shape[:-3] = 1
            mask_shape = [1] * len(shape)
            mask_shape[-2] = num_cols
            mask = torch.from_numpy(mask.reshape(*mask_shape).astype(np.float32))
            mask = mask.reshape(*mask_shape)
            mask[:, :, :acq_start] = 0
            mask[:, :, acq_end:] = 0

        return (
            masked_kspace,
            mask.byte(),
            target,
            fname,
            slice_num,
            max_value,
            crop_size,
        )


class LieFcDataTransform:
    """
    Data Transformer for training U-Net models.
    """

    def __init__(
            self,
            which_challenge: str,
            mask_func: Optional[MaskFunc] = None,
            use_seed: bool = True,
    ):
        """
        Args:
            which_challenge: Challenge from ("singlecoil", "multicoil").
            mask_func: Optional; A function that can create a mask of
                appropriate shape.
            use_seed: If true, this class computes a pseudo random number
                generator seed from the filename. This ensures that the same
                mask is used for all the slices of a given volume every time.
        """
        if which_challenge not in ("singlecoil", "multicoil"):
            raise ValueError("Challenge should either be 'singlecoil' or 'multicoil'")

        self.mask_func = mask_func
        self.which_challenge = which_challenge
        self.use_seed = use_seed
        self.shape = [320, 320, 2]
        mask_func_ = create_mask_for_mask_type('equispaced', [0.08], [4])
        seed_ = (102, 105, 108, 101, 49, 48, 48, 50, 51, 56, 48, 46, 104, 53)
        self.mask = mask_func_(self.shape, seed_)
        # print('')

    def __call__(
            self,
            kspace: np.ndarray,
            mask: np.ndarray,
            target: np.ndarray,
            attrs: Dict,
            fname: str,
            slice_num: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str, int, float]:
        """
        Args:
            kspace: Input k-space of shape (num_coils, rows, cols) for
                multi-coil data or (rows, cols) for single coil data.
            mask: Mask from the test dataset.
            target: Target image.
            attrs: Acquisition related information stored in the HDF5 object.
            fname: File name.
            slice_num: Serial number of the slice.

        Returns:
            tuple containing:
                image: Zero-filled input image.
                target: Target image converted to a torch.Tensor.
                mean: Mean value used for normalization.
                std: Standard deviation value used for normalization.
                fname: File name.
                slice_num: Serial number of the slice.
        """
        kspace = to_tensor(kspace)

        # check for max value
        max_value = attrs["max"] if "max" in attrs.keys() else 0.0

        # # apply mask
        # if self.mask_func:
        #     seed = None if not self.use_seed else tuple(map(ord, fname))
        #     masked_kspace, mask = apply_mask(kspace, self.mask_func, seed)
        # else:
        #     masked_kspace = kspace

        # # inverse Fourier transform to get zero filled solution
        # image = fastmri.ifft2c(masked_kspace)

        full_image = fastmri.ifft2c(kspace)
        full_image_320 = complex_center_crop(full_image, (320, 320))
        kspace = fastmri.fft2c(full_image_320)

        # apply mask
        masked_kspace = kspace * self.mask + 0.0

        # visual code.
        # from matplotlib import pyplot as plt
        # fade_kspace = torch.ones((320,320,1))
        # fade_kspace = fade_kspace * self.mask + 0.0
        # plt.imshow(fade_kspace[...,0].numpy(),'gray')
        # plt.savefig('/home/vpa/test.png')

        # normalize input
        kspace = kspace.permute(2, 0, 1)
        masked_kspace = masked_kspace.permute(2, 0, 1)
        kspace, mean, std = normalize_instance(kspace, eps=1e-11)
        masked_kspace = normalize(masked_kspace, mean, std, eps=1e-11)

        return masked_kspace, kspace, mean, std, fname, slice_num, max_value


class LieFc_With_Unet_DataTransform:
    """
    Data Transformer for training U-Net models.
    """

    def __init__(
            self,
            which_challenge: str,
            mask_func: Optional[MaskFunc] = None,
            use_seed: bool = True,
    ):
        """
        Args:
            which_challenge: Challenge from ("singlecoil", "multicoil").
            mask_func: Optional; A function that can create a mask of
                appropriate shape.
            use_seed: If true, this class computes a pseudo random number
                generator seed from the filename. This ensures that the same
                mask is used for all the slices of a given volume every time.
        """
        if which_challenge not in ("singlecoil", "multicoil"):
            raise ValueError("Challenge should either be 'singlecoil' or 'multicoil'")

        self.mask_func = mask_func
        self.which_challenge = which_challenge
        self.use_seed = use_seed
        self.shape = [320, 320, 2]
        mask_func_ = create_mask_for_mask_type('equispaced', [0.08], [4])
        seed_ = (102, 105, 108, 101, 49, 48, 48, 50, 51, 56, 48, 46, 104, 53)
        self.mask = mask_func_(self.shape, seed_)
        # print('')

    def __call__(
            self,
            kspace: np.ndarray,
            mask: np.ndarray,
            target: np.ndarray,
            attrs: Dict,
            fname: str,
            slice_num: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str, int, float]:
        """
        Args:
            kspace: Input k-space of shape (num_coils, rows, cols) for
                multi-coil data or (rows, cols) for single coil data.
            mask: Mask from the test dataset.
            target: Target image.
            attrs: Acquisition related information stored in the HDF5 object.
            fname: File name.
            slice_num: Serial number of the slice.

        Returns:
            tuple containing:
                image: Zero-filled input image.
                target: Target image converted to a torch.Tensor.
                mean: Mean value used for normalization.
                std: Standard deviation value used for normalization.
                fname: File name.
                slice_num: Serial number of the slice.
        """
        kspace = to_tensor(kspace)

        # check for max value
        max_value = attrs["max"] if "max" in attrs.keys() else 0.0

        # # apply mask
        # if self.mask_func:
        #     seed = None if not self.use_seed else tuple(map(ord, fname))
        #     masked_kspace, mask = apply_mask(kspace, self.mask_func, seed)
        # else:
        #     masked_kspace = kspace

        # # inverse Fourier transform to get zero filled solution
        # image = fastmri.ifft2c(masked_kspace)

        full_image = fastmri.ifft2c(kspace)
        full_image_320 = complex_center_crop(full_image, (320, 320))
        kspace = fastmri.fft2c(full_image_320)

        # apply mask
        masked_kspace = kspace * self.mask + 0.0

        # visual code.
        # from matplotlib import pyplot as plt
        # fade_kspace = torch.ones((320,320,1))
        # fade_kspace = fade_kspace * self.mask + 0.0
        # plt.imshow(fade_kspace[...,0].numpy(),'gray')
        # plt.savefig('/home/vpa/test.png')

        # normalize input
        kspace = kspace.permute(2, 0, 1)
        masked_kspace = masked_kspace.permute(2, 0, 1)
        kspace, mean, std = normalize_instance(kspace, eps=1e-11)
        masked_kspace = normalize(masked_kspace, mean, std, eps=1e-11)
        # normalize target
        if target is not None:
            target = to_tensor(target)
            target = center_crop(target, (320, 320))
            target, mean_img, std_img = normalize_instance(target, eps=1e-11)
            # target = normalize(target, mean, std, eps=1e-11)
            target = target.clamp(-6, 6)
        else:
            target = torch.Tensor([0])

        return masked_kspace, kspace, mean, std, fname, slice_num, max_value, target, mean_img, std_img


class Unet_With_High_Freq_DataTransform:
    """
    Data Transformer for training U-Net models.
    """

    def __init__(
            self,
            which_challenge: str,
            mask_func: Optional[MaskFunc] = None,
            use_seed: bool = True,
    ):
        """
        Args:
            which_challenge: Challenge from ("singlecoil", "multicoil").
            mask_func: Optional; A function that can create a mask of
                appropriate shape.
            use_seed: If true, this class computes a pseudo random number
                generator seed from the filename. This ensures that the same
                mask is used for all the slices of a given volume every time.
        """
        if which_challenge not in ("singlecoil", "multicoil"):
            raise ValueError("Challenge should either be 'singlecoil' or 'multicoil'")

        self.mask_func = mask_func
        self.which_challenge = which_challenge
        self.use_seed = use_seed

    def __call__(
            self,
            kspace: np.ndarray,
            mask: np.ndarray,
            target: np.ndarray,
            attrs: Dict,
            fname: str,
            slice_num: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str, int, float]:
        """
        Args:
            kspace: Input k-space of shape (num_coils, rows, cols) for
                multi-coil data or (rows, cols) for single coil data.
            mask: Mask from the test dataset.
            target: Target image.
            attrs: Acquisition related information stored in the HDF5 object.
            fname: File name.
            slice_num: Serial number of the slice.

        Returns:
            tuple containing:
                image: Zero-filled input image.
                target: Target image converted to a torch.Tensor.
                mean: Mean value used for normalization.
                std: Standard deviation value used for normalization.
                fname: File name.
                slice_num: Serial number of the slice.
        """
        kspace = to_tensor(kspace)

        # check for max value
        max_value = attrs["max"] if "max" in attrs.keys() else 0.0

        # apply mask
        if self.mask_func:
            seed = None if not self.use_seed else tuple(map(ord, fname))
            masked_kspace, mask = apply_mask(kspace, self.mask_func, seed)
        else:
            masked_kspace = kspace

        # inverse Fourier transform to get zero filled solution
        image = fastmri.ifft2c(masked_kspace)

        # normalize input
        image = image.permute(2, 0, 1)

        # normalize target
        if target is not None:
            target = to_tensor(target)
        else:
            target = torch.Tensor([0])
        d_matrix = self.make_transform_matrix(20, (image.shape))
        high_pass_kspace = kspace * d_matrix
        high_pass_img = center_crop(fastmri.complex_abs(fastmri.ifft2c(high_pass_kspace)), (320, 320))
        # import matplotlib.pyplot as plt
        # plt.imshow(high_pass_img, 'gray')
        # plt.savefig('/raid/MRI_group/test.png')
        return image, target, 0, 0, fname, slice_num, max_value, high_pass_img

    def make_transform_matrix(self, d, img_size):
        img_temp = torch.zeros(img_size)
        hangshu = torch.arange(0, img_size[1], 1)
        lieshu = torch.arange(0, img_size[2], 1)
        for i in range(img_size[2]):
            img_temp[0, :, i] = hangshu
        for i in range(img_size[1]):
            img_temp[1, i, :] = lieshu
        hangshu_mid = (img_size[1] - 1) / 2
        lieshu_mid = (img_size[2] - 1) / 2
        img_temp[0] -= hangshu_mid
        img_temp[1] -= lieshu_mid
        dis = torch.sqrt(img_temp[0] ** 2 + img_temp[1] ** 2)
        transfor_matrix = (dis >= d)
        img_temp[0] = transfor_matrix
        img_temp[1] = transfor_matrix
        return img_temp.permute(1, 2, 0)


class K_UnetDataTransform:
    """
    Data Transformer for training U-Net models.
    """

    def __init__(
            self,
            which_challenge: str,
            mask_func: Optional[MaskFunc] = None,
            use_seed: bool = True,
    ):
        """
        Args:
            which_challenge: Challenge from ("singlecoil", "multicoil").
            mask_func: Optional; A function that can create a mask of
                appropriate shape.
            use_seed: If true, this class computes a pseudo random number
                generator seed from the filename. This ensures that the same
                mask is used for all the slices of a given volume every time.
        """
        if which_challenge not in ("singlecoil", "multicoil"):
            raise ValueError("Challenge should either be 'singlecoil' or 'multicoil'")

        self.mask_func = mask_func
        self.which_challenge = which_challenge
        self.use_seed = use_seed

    def __call__(
            self,
            kspace: np.ndarray,
            mask: np.ndarray,
            target: np.ndarray,
            attrs: Dict,
            fname: str,
            slice_num: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str, int, float]:
        """
        Args:
            kspace: Input k-space of shape (num_coils, rows, cols) for
                multi-coil data or (rows, cols) for single coil data.
            mask: Mask from the test dataset.
            target: Target image.
            attrs: Acquisition related information stored in the HDF5 object.
            fname: File name.
            slice_num: Serial number of the slice.

        Returns:
            tuple containing:
                image: Zero-filled input image.
                target: Target image converted to a torch.Tensor.
                mean: Mean value used for normalization.
                std: Standard deviation value used for normalization.
                fname: File name.
                slice_num: Serial number of the slice.
        """
        kspace = to_tensor(kspace)

        # check for max value
        max_value = attrs["max"] if "max" in attrs.keys() else 0.0

        # apply mask
        if self.mask_func:
            seed = None if not self.use_seed else tuple(map(ord, fname))
            masked_kspace, mask = apply_mask(kspace, self.mask_func, seed)
        else:
            masked_kspace = kspace

        # inverse Fourier transform to get zero filled solution
        # image = fastmri.ifft2c(masked_kspace)

        # crop input to correct size
        # if target is not None:
        #     crop_size = (target.shape[-2], target.shape[-1])
        # else:
        #     crop_size = (attrs["recon_size"][0], attrs["recon_size"][1])

        # check for FLAIR 203
        # if image.shape[-2] < crop_size[1]:
        #     crop_size = (image.shape[-2], image.shape[-2])

        # image = complex_center_crop(image, crop_size)

        # absolute value
        # image = fastmri.complex_abs(image)

        # apply Root-Sum-of-Squares if multicoil data
        # if self.which_challenge == "multicoil":
        #     image = fastmri.rss(image)

        # normalize input
        masked_kspace = masked_kspace.permute(2, 0, 1)
        # image, mean, std = self.norm(image)

        # normalize target
        if target is not None:
            target = to_tensor(target)
            target_max = torch.max(target)
            target_min = torch.min(target)
            target_guiyi = ((target - target_min) / target_max).numpy()
            target_guiyi = torch.from_numpy(guideFilter(target_guiyi, target_guiyi, (16, 16), 0.01, 0.5)).unsqueeze(
                0) * target_max + target_min
        else:
            target = torch.Tensor([0])

        return masked_kspace, target, 0, 0, fname, slice_num, max_value, mask.permute(2, 0, 1).byte(), target_guiyi


class ImageUnetWithTargetKDataTransform:
    """
    Data Transformer for training U-Net models.
    """

    def __init__(
            self,
            which_challenge: str,
            mask_func: Optional[MaskFunc] = None,
            use_seed: bool = True,
    ):
        """
        Args:
            which_challenge: Challenge from ("singlecoil", "multicoil").
            mask_func: Optional; A function that can create a mask of
                appropriate shape.
            use_seed: If true, this class computes a pseudo random number
                generator seed from the filename. This ensures that the same
                mask is used for all the slices of a given volume every time.
        """
        if which_challenge not in ("singlecoil", "multicoil"):
            raise ValueError("Challenge should either be 'singlecoil' or 'multicoil'")

        self.mask_func = mask_func
        self.which_challenge = which_challenge
        self.use_seed = use_seed

    def __call__(
            self,
            kspace: np.ndarray,
            mask: np.ndarray,
            target: np.ndarray,
            attrs: Dict,
            fname: str,
            slice_num: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str, int, float]:
        """
        Args:
            kspace: Input k-space of shape (num_coils, rows, cols) for
                multi-coil data or (rows, cols) for single coil data.
            mask: Mask from the test dataset.
            target: Target image.
            attrs: Acquisition related information stored in the HDF5 object.
            fname: File name.
            slice_num: Serial number of the slice.

        Returns:
            tuple containing:
                image: Zero-filled input image.
                target: Target image converted to a torch.Tensor.
                mean: Mean value used for normalization.
                std: Standard deviation value used for normalization.
                fname: File name.
                slice_num: Serial number of the slice.
        """
        kspace = to_tensor(kspace)

        # check for max value
        max_value = attrs["max"] if "max" in attrs.keys() else 0.0

        # apply mask
        if self.mask_func:
            seed = None if not self.use_seed else tuple(map(ord, fname))
            masked_kspace, mask = apply_mask(kspace, self.mask_func, seed)
        else:
            masked_kspace = kspace

        # inverse Fourier transform to get zero filled solution
        image = fastmri.ifft2c(masked_kspace)

        # crop input to correct size
        # if target is not None:
        #     crop_size = (target.shape[-2], target.shape[-1])
        # else:
        #     crop_size = (attrs["recon_size"][0], attrs["recon_size"][1])

        # check for FLAIR 203
        # if image.shape[-2] < crop_size[1]:
        #     crop_size = (image.shape[-2], image.shape[-2])

        # image = complex_center_crop(image, crop_size)

        # absolute value
        # image = fastmri.complex_abs(image)

        # apply Root-Sum-of-Squares if multicoil data
        # if self.which_challenge == "multicoil":
        #     image = fastmri.rss(image)

        # normalize input
        image = image.permute(2, 0, 1)
        # kspace = kspace.permute(2,0,1)
        # mask = mask.permute(2,0,1)
        # image, mean, std = self.norm(image)

        # normalize target
        if target is not None:
            target = to_tensor(target)
            # target = center_crop(target, crop_size)
            # target = normalize(target, mean, std, eps=1e-11)
            # target, mean_target, std_target = normalize_instance(target, eps=1e-11)
            # target = target.clamp(-6, 6)
        else:
            target = torch.Tensor([0])

        return image, target, 0, 0, fname, slice_num, max_value, kspace, mask


class ImageUnetWithTargetKData320320Transform:
    """
    Data Transformer for training U-Net models.
    """

    def __init__(
            self,
            which_challenge: str,
            mask_func: Optional[MaskFunc] = None,
            use_seed: bool = True,
    ):
        """
        Args:
            which_challenge: Challenge from ("singlecoil", "multicoil").
            mask_func: Optional; A function that can create a mask of
                appropriate shape.
            use_seed: If true, this class computes a pseudo random number
                generator seed from the filename. This ensures that the same
                mask is used for all the slices of a given volume every time.
        """
        if which_challenge not in ("singlecoil", "multicoil"):
            raise ValueError("Challenge should either be 'singlecoil' or 'multicoil'")

        self.mask_func = mask_func
        self.which_challenge = which_challenge
        self.use_seed = use_seed
        self.shape = [320, 320, 2]
        mask_func_ = create_mask_for_mask_type('equispaced', [0.08], [4])
        seed_ = (102, 105, 108, 101, 49, 48, 48, 50, 51, 56, 48, 46, 104, 53)
        self.mask = mask_func_(self.shape, seed_)

    def __call__(
            self,
            kspace: np.ndarray,
            mask: np.ndarray,
            target: np.ndarray,
            attrs: Dict,
            fname: str,
            slice_num: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str, int, float]:
        """
        Args:
            kspace: Input k-space of shape (num_coils, rows, cols) for
                multi-coil data or (rows, cols) for single coil data.
            mask: Mask from the test dataset.
            target: Target image.
            attrs: Acquisition related information stored in the HDF5 object.
            fname: File name.
            slice_num: Serial number of the slice.

        Returns:
            tuple containing:
                image: Zero-filled input image.
                target: Target image converted to a torch.Tensor.
                mean: Mean value used for normalization.
                std: Standard deviation value used for normalization.
                fname: File name.
                slice_num: Serial number of the slice.
        """
        kspace = to_tensor(kspace)

        # check for max value
        max_value = attrs["max"] if "max" in attrs.keys() else 0.0
        full_image = fastmri.ifft2c(kspace)
        full_image_320 = complex_center_crop(full_image, (320, 320))
        kspace = fastmri.fft2c(full_image_320)

        # apply mask
        # if self.mask_func:
        #     seed = None if not self.use_seed else tuple(map(ord, fname))
        #     masked_kspace, mask = apply_mask(kspace, self.mask_func, seed)
        # else:
        #     masked_kspace = kspace
        masked_kspace = kspace * self.mask + 0.0

        # inverse Fourier transform to get zero filled solution
        image = fastmri.ifft2c(masked_kspace)

        # crop input to correct size
        # if target is not None:
        #     crop_size = (target.shape[-2], target.shape[-1])
        # else:
        #     crop_size = (attrs["recon_size"][0], attrs["recon_size"][1])

        # check for FLAIR 203
        # if image.shape[-2] < crop_size[1]:
        #     crop_size = (image.shape[-2], image.shape[-2])

        # image = complex_center_crop(image, crop_size)

        # absolute value
        # image = fastmri.complex_abs(image)

        # apply Root-Sum-of-Squares if multicoil data
        # if self.which_challenge == "multicoil":
        #     image = fastmri.rss(image)

        # normalize input
        image = image.permute(2, 0, 1)
        # kspace = kspace.permute(2,0,1)
        # mask = mask.permute(2,0,1)
        # image, mean, std = self.norm(image)

        # normalize target
        if target is not None:
            target = to_tensor(target)
            # target = center_crop(target, crop_size)
            # target = normalize(target, mean, std, eps=1e-11)
            # target, mean_target, std_target = normalize_instance(target, eps=1e-11)
            # target = target.clamp(-6, 6)
        else:
            target = torch.Tensor([0])

        return image, target, 0, 0, fname, slice_num, max_value, kspace, self.mask


class ImageUnetWithTargetKData41_320320Transform:
    """
    Data Transformer for training U-Net models.
    """

    def __init__(
            self,
            which_challenge: str,
            mask_func: Optional[MaskFunc] = None,
            use_seed: bool = True,
    ):
        """
        Args:
            which_challenge: Challenge from ("singlecoil", "multicoil").
            mask_func: Optional; A function that can create a mask of
                appropriate shape.
            use_seed: If true, this class computes a pseudo random number
                generator seed from the filename. This ensures that the same
                mask is used for all the slices of a given volume every time.
        """
        if which_challenge not in ("singlecoil", "multicoil"):
            raise ValueError("Challenge should either be 'singlecoil' or 'multicoil'")

        self.mask_func = mask_func
        self.which_challenge = which_challenge
        self.use_seed = use_seed
        self.shape = [320, 320, 2]
        mask_func_ = create_mask_for_mask_type('equispaced', [0.08], [4])
        seed_ = (102, 105, 108, 101, 49, 48, 48, 50, 51, 56, 48, 46, 104, 53)
        self.mask = 1 - mask_func_(self.shape, seed_)
        self.mask[:, 160 - 13:160 + 13, :] = 1

    def __call__(
            self,
            kspace: np.ndarray,
            mask: np.ndarray,
            target: np.ndarray,
            attrs: Dict,
            fname: str,
            slice_num: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str, int, float]:
        """
        Args:
            kspace: Input k-space of shape (num_coils, rows, cols) for
                multi-coil data or (rows, cols) for single coil data.
            mask: Mask from the test dataset.
            target: Target image.
            attrs: Acquisition related information stored in the HDF5 object.
            fname: File name.
            slice_num: Serial number of the slice.

        Returns:
            tuple containing:
                image: Zero-filled input image.
                target: Target image converted to a torch.Tensor.
                mean: Mean value used for normalization.
                std: Standard deviation value used for normalization.
                fname: File name.
                slice_num: Serial number of the slice.
        """
        kspace = to_tensor(kspace)

        # check for max value
        max_value = attrs["max"] if "max" in attrs.keys() else 0.0
        full_image = fastmri.ifft2c(kspace)
        full_image_320 = complex_center_crop(full_image, (320, 320))
        kspace = fastmri.fft2c(full_image_320)

        # apply mask
        # if self.mask_func:
        #     seed = None if not self.use_seed else tuple(map(ord, fname))
        #     masked_kspace, mask = apply_mask(kspace, self.mask_func, seed)
        # else:
        #     masked_kspace = kspace
        masked_kspace = kspace * self.mask + 0.0

        # inverse Fourier transform to get zero filled solution
        image = fastmri.ifft2c(masked_kspace)

        # crop input to correct size
        # if target is not None:
        #     crop_size = (target.shape[-2], target.shape[-1])
        # else:
        #     crop_size = (attrs["recon_size"][0], attrs["recon_size"][1])

        # check for FLAIR 203
        # if image.shape[-2] < crop_size[1]:
        #     crop_size = (image.shape[-2], image.shape[-2])

        # image = complex_center_crop(image, crop_size)

        # absolute value
        # image = fastmri.complex_abs(image)

        # apply Root-Sum-of-Squares if multicoil data
        # if self.which_challenge == "multicoil":
        #     image = fastmri.rss(image)

        # normalize input
        image = image.permute(2, 0, 1)
        # kspace = kspace.permute(2,0,1)
        # mask = mask.permute(2,0,1)
        # image, mean, std = self.norm(image)

        # normalize target
        if target is not None:
            target = to_tensor(target)
            # target = center_crop(target, crop_size)
            # target = normalize(target, mean, std, eps=1e-11)
            # target, mean_target, std_target = normalize_instance(target, eps=1e-11)
            # target = target.clamp(-6, 6)
        else:
            target = torch.Tensor([0])

        return image, target, 0, 0, fname, slice_num, max_value, kspace, self.mask


class ImageUnetWithTargetKData41_320320Transform_multicoils:
    """
    Data Transformer for training U-Net models.
    """

    def __init__(
            self,
            which_challenge: str,
            mask_func: Optional[MaskFunc] = None,
            use_seed: bool = True,
    ):
        """
        Args:
            which_challenge: Challenge from ("singlecoil", "multicoil").
            mask_func: Optional; A function that can create a mask of
                appropriate shape.
            use_seed: If true, this class computes a pseudo random number
                generator seed from the filename. This ensures that the same
                mask is used for all the slices of a given volume every time.
        """
        if which_challenge not in ("singlecoil", "multicoil"):
            raise ValueError("Challenge should either be 'singlecoil' or 'multicoil'")

        self.mask_func = mask_func
        self.which_challenge = which_challenge
        self.use_seed = use_seed
        self.shape = [22, 640, 320, 2]
        mask_func_ = create_mask_for_mask_type('equispaced', [0.08], [4])
        seed_ = (102, 105, 108, 101, 49, 48, 48, 50, 51, 56, 48, 46, 104, 53)
        self.mask = mask_func_(self.shape, seed_)
        self.except_mid_mask = torch.zeros_like(self.mask)
        self.except_mid_mask[:, :, 160 - 13:160 + 13, :] = 1
        self.max_coils = 22

    def __call__(
            self,
            kspace: np.ndarray,
            mask: np.ndarray,
            target: np.ndarray,
            attrs: Dict,
            fname: str,
            slice_num: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str, int, float]:
        """
        Args:
            kspace: Input k-space of shape (num_coils, rows, cols) for
                multi-coil data or (rows, cols) for single coil data.
            mask: Mask from the test dataset.
            target: Target image.
            attrs: Acquisition related information stored in the HDF5 object.
            fname: File name.
            slice_num: Serial number of the slice.

        Returns:
            tuple containing:
                image: Zero-filled input image.
                target: Target image converted to a torch.Tensor.
                mean: Mean value used for normalization.
                std: Standard deviation value used for normalization.
                fname: File name.
                slice_num: Serial number of the slice.
        """
        coils = kspace.shape[0]
        if coils > self.max_coils:
            self.max_coils = coils
            # print(self.max_coils)
        kspace = to_tensor(kspace)
        if coils < self.max_coils:
            gap = self.max_coils - coils
            if gap <= coils:
                kspace = torch.cat([kspace, kspace[:gap]], 0)
            else:
                while (kspace.shape[0] < self.max_coils):
                    kspace = torch.cat([kspace, kspace], 0)
                kspace = kspace[:self.max_coils]

        # check for max value
        max_value = attrs["max"] if "max" in attrs.keys() else 0.0
        # full_image = fastmri.ifft2c(kspace)
        # full_image_320 = complex_center_crop(full_image, (320,320))
        # kspace = fastmri.fft2c(full_image_320)

        # apply mask
        # if self.mask_func:
        #     seed = None if not self.use_seed else tuple(map(ord, fname))
        #     masked_kspace, mask = apply_mask(kspace, self.mask_func, seed)
        # else:
        #     masked_kspace = kspace
        masked_kspace = kspace * self.mask + 0.0

        # inverse Fourier transform to get zero filled solution
        image = fastmri.ifft2c(masked_kspace)

        # crop input to correct size
        # if target is not None:
        #     crop_size = (target.shape[-2], target.shape[-1])
        # else:
        #     crop_size = (attrs["recon_size"][0], attrs["recon_size"][1])

        # check for FLAIR 203
        # if image.shape[-2] < crop_size[1]:
        #     crop_size = (image.shape[-2], image.shape[-2])

        # image = complex_center_crop(image, crop_size)

        # absolute value
        # image = fastmri.complex_abs(image)

        # apply Root-Sum-of-Squares if multicoil data
        # if self.which_challenge == "multicoil":
        #     image = fastmri.rss(image)

        # normalize input
        image = self.complex_to_chan_dim(image.unsqueeze(0))[0]
        # kspace = kspace.permute(2,0,1)
        # mask = mask.permute(2,0,1)
        # image, mean, std = self.norm(image)

        # normalize target
        # if target is not None:
        #     target = to_tensor(target)
        # target = center_crop(target, crop_size)
        # target = normalize(target, mean, std, eps=1e-11)
        # target, mean_target, std_target = normalize_instance(target, eps=1e-11)
        # target = target.clamp(-6, 6)
        # else:
        #     target = torch.Tensor([0])
        # if image.shape[0] != 44:
        #     print('')
        target = center_crop(fastmri.complex_abs((fastmri.ifft2c(kspace))), [320, 320])
        target = fastmri.rss(target)

        return image, target, 0, 0, fname, slice_num, max_value, kspace, self.mask, self.except_mid_mask

    def complex_to_chan_dim(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w, two = x.shape
        assert two == 2
        return x.permute(0, 4, 1, 2, 3).reshape(b, 2 * c, h, w)

    def chan_complex_to_last_dim(self, x: torch.Tensor) -> torch.Tensor:
        b, c2, h, w = x.shape
        assert c2 % 2 == 0
        c = c2 // 2
        return x.view(b, 2, c, h, w).permute(0, 2, 3, 4, 1).contiguous()


class UnetData_2channel_allresolution_Transform:
    """
    Data Transformer for training U-Net models.
    """

    def __init__(
            self,
            which_challenge: str,
            mask_func: Optional[MaskFunc] = None,
            use_seed: bool = True,
    ):
        """
        Args:
            which_challenge: Challenge from ("singlecoil", "multicoil").
            mask_func: Optional; A function that can create a mask of
                appropriate shape.
            use_seed: If true, this class computes a pseudo random number
                generator seed from the filename. This ensures that the same
                mask is used for all the slices of a given volume every time.
        """
        if which_challenge not in ("singlecoil", "multicoil"):
            raise ValueError("Challenge should either be 'singlecoil' or 'multicoil'")

        self.mask_func = mask_func
        self.which_challenge = which_challenge
        self.use_seed = use_seed

    def __call__(
            self,
            kspace: np.ndarray,
            mask: np.ndarray,
            target: np.ndarray,
            attrs: Dict,
            fname: str,
            slice_num: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str, int, float]:
        """
        Args:
            kspace: Input k-space of shape (num_coils, rows, cols) for
                multi-coil data or (rows, cols) for single coil data.
            mask: Mask from the test dataset.
            target: Target image.
            attrs: Acquisition related information stored in the HDF5 object.
            fname: File name.
            slice_num: Serial number of the slice.

        Returns:
            tuple containing:
                image: Zero-filled input image.
                target: Target image converted to a torch.Tensor.
                mean: Mean value used for normalization.
                std: Standard deviation value used for normalization.
                fname: File name.
                slice_num: Serial number of the slice.
        """
        kspace = to_tensor(kspace)

        # check for max value
        max_value = attrs["max"] if "max" in attrs.keys() else 0.0

        # apply mask
        if self.mask_func:
            seed = None if not self.use_seed else tuple(map(ord, fname))
            masked_kspace, mask = apply_mask(kspace, self.mask_func, seed)
        else:
            masked_kspace = kspace

        # inverse Fourier transform to get zero filled solution
        image = fastmri.ifft2c(masked_kspace).permute(2, 0, 1)

        # normalize target
        if target is not None:
            target = to_tensor(target)
        else:
            target = torch.Tensor([0])

        return image, target, 0, 0, fname, slice_num, max_value, mask.byte(), masked_kspace


class UnetData_2channel_allresolution_Transform_visual:
    """
    Data Transformer for training U-Net models.
    """

    def __init__(
            self,
            which_challenge: str,
            mask_func: Optional[MaskFunc] = None,
            use_seed: bool = True,
    ):
        """
        Args:
            which_challenge: Challenge from ("singlecoil", "multicoil").
            mask_func: Optional; A function that can create a mask of
                appropriate shape.
            use_seed: If true, this class computes a pseudo random number
                generator seed from the filename. This ensures that the same
                mask is used for all the slices of a given volume every time.
        """
        if which_challenge not in ("singlecoil", "multicoil"):
            raise ValueError("Challenge should either be 'singlecoil' or 'multicoil'")

        self.mask_func = mask_func
        self.which_challenge = which_challenge
        self.use_seed = use_seed

    def __call__(
            self,
            kspace: np.ndarray,
            mask: np.ndarray,
            target: np.ndarray,
            attrs: Dict,
            fname: str,
            slice_num: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str, int, float]:
        """
        Args:
            kspace: Input k-space of shape (num_coils, rows, cols) for
                multi-coil data or (rows, cols) for single coil data.
            mask: Mask from the test dataset.
            target: Target image.
            attrs: Acquisition related information stored in the HDF5 object.
            fname: File name.
            slice_num: Serial number of the slice.

        Returns:
            tuple containing:
                image: Zero-filled input image.
                target: Target image converted to a torch.Tensor.
                mean: Mean value used for normalization.
                std: Standard deviation value used for normalization.
                fname: File name.
                slice_num: Serial number of the slice.
        """
        # import h5py
        # hf = h5py.File('/raid/dataset/singlecoil_knee_wyz/singlecoil_val/file1001344.h5')
        # kspace = hf['kspace'][15]
        # # print(list(hf.keys()))
        # target = hf['reconstruction_esc'][15]
        # print(kspace.shape)
        kspace = to_tensor(kspace)  # 1,640,368
        # num_cols = kspace.shape[-1]
        # num_low_freqs = int(round(num_cols * 0.08))
        # pad = (num_cols - num_low_freqs + 1) // 2
        #
        # kspace = torch.rot90(kspace[:, :, 0:pad - 1], 2)

        # check for max value
        max_value = attrs["max"] if "max" in attrs.keys() else 0.0

        # apply mask
        if self.mask_func:
            seed = None if not self.use_seed else tuple(map(ord, fname))
            masked_kspace, mask,_ = apply_mask(kspace, self.mask_func, seed=seed)
        else:
            masked_kspace = kspace
            mask = torch.from_numpy(mask).float().unsqueeze(1).unsqueeze(0)

        # plt.imshow(torch.log(fastmri.complex_abs(masked_kspace)+ 0.000000001), cmap='gray')
        # plt.show()

        # masked_kspace_copyy = copy.deepcopy(masked_kspace)
        # masked_kspace = masked_kspace.unsqueeze(dim=0)
        # mask = mask.unsqueeze(dim=0)
        # masked_kspace = apply_grappa(masked_kspace, mask).squeeze(dim=0)
        # masked_kspace = masked_kspace_copyy + (masked_kspace_copyy==0)*masked_kspace
        # mask = mask.squeeze(dim=0)

        masked_kspace_copyy = copy.deepcopy(masked_kspace)
        masked_kspace_copy = copy.deepcopy(masked_kspace)


        image_org = fastmri.ifft2c(masked_kspace_copyy).permute(2,0,1)
        image = image_org


        # normalize target
        if target is not None:
            target = to_tensor(target)
        else:
            target = torch.Tensor([0])

        return image, target, 0, 0, fname, slice_num, max_value, mask.byte(), masked_kspace_copyy,image_org


class UnetData_2channel_allresolution_Transform_md:
    """
    Data Transformer for training U-Net models.
    """

    def __init__(
            self,
            which_challenge: str,
            mask_func: Optional[MaskFunc] = None,
            use_seed: bool = True,
    ):
        """
        Args:
            which_challenge: Challenge from ("singlecoil", "multicoil").
            mask_func: Optional; A function that can create a mask of
                appropriate shape.
            use_seed: If true, this class computes a pseudo random number
                generator seed from the filename. This ensures that the same
                mask is used for all the slices of a given volume every time.
        """
        if which_challenge not in ("singlecoil", "multicoil"):
            raise ValueError("Challenge should either be 'singlecoil' or 'multicoil'")

        self.mask_func = mask_func
        self.which_challenge = which_challenge
        self.use_seed = use_seed

    def __call__(
            self,
            kspace: np.ndarray,
            mask: np.ndarray,
            target: np.ndarray,
            attrs: Dict,
            fname: str,
            slice_num: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str, int, float]:
        """
        Args:
            kspace: Input k-space of shape (num_coils, rows, cols) for
                multi-coil data or (rows, cols) for single coil data.
            mask: Mask from the test dataset.
            target: Target image.
            attrs: Acquisition related information stored in the HDF5 object.
            fname: File name.
            slice_num: Serial number of the slice.

        Returns:
            tuple containing:
                image: Zero-filled input image.
                target: Target image converted to a torch.Tensor.
                mean: Mean value used for normalization.
                std: Standard deviation value used for normalization.
                fname: File name.
                slice_num: Serial number of the slice.
        """
        kspace = to_tensor(kspace)
        image = fastmri.ifft2c(kspace)
        image_j = image[..., 0] + 1j * image[..., 1]

        norm = torch.abs(image_j)
        min = torch.min(norm)
        max = torch.max(norm)

        kspace = fastmri.fft2c(to_tensor((image_j - min) / (max - min) * 255))

        # check for max value
        max_value = attrs["max"] if "max" in attrs.keys() else 0.0

        # apply mask
        if self.mask_func:
            seed = None if not self.use_seed else tuple(map(ord, fname))
            masked_kspace, mask = apply_mask(kspace, self.mask_func, seed)
        else:
            masked_kspace = kspace

        # inverse Fourier transform to get zero filled solution
        image = fastmri.ifft2c(masked_kspace).permute(2, 0, 1)

        # normalize target
        if target is not None:
            target = to_tensor(target)
            target = (target - min) / (max - min) * 255
        else:
            target = torch.Tensor([0])

        return image, target, min, max, fname, slice_num, max_value, mask.byte(), masked_kspace
