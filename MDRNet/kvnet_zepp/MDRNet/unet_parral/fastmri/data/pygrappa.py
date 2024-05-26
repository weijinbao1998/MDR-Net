import numpy as np
import transforms as T

import  torch
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

def kspace_to_image(slice_kspace):
    """
    Transfroms kspace data to real-valued image.
    Args:
        slice_kspace (numpy.array): Multi-coil input k-space of shape (num_coils, rows, cols)
    Returns:
        image: (numpy.array): Multi-channel image of shape (num_coils, rows, cols)
    """
    slice_kspace2 = T.to_tensor(slice_kspace)  # Convert from numpy array to pytorch tensor
    slice_image = T.ifft2(slice_kspace2)  # Apply Inverse Fourier Transform to get the complex image
    slice_image_abs = T.complex_abs(slice_image)  # Compute absolute value to get a real image
    return slice_image_abs


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
                                        kernel_size=(3, 3), coil_axis=0)
    return T.to_tensor(preprocessed_masked_kspace)

def to_tensor(data):
    """
    Convert numpy array to PyTorch tensor. For complex arrays, the real and imaginary parts
    are stacked along the last dimension.
    Args:
        data (np.array): Input numpy array
    Returns:
        torch.Tensor: PyTorch version of data
    """
    if np.iscomplexobj(data):
        data = np.stack((data.real, data.imag), axis=-1)
    return torch.Tensor(data)


def tensor_to_complex_np(data):
    """
    Converts a complex torch tensor to numpy array.
    Args:
        data (torch.Tensor): Input data to be converted to numpy.

    Returns:
        np.array: Complex numpy version of data
    """
    data = data.numpy()
    return data[..., 0] + 1j * data[..., 1]

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