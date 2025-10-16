"""
Conjugate gradient mapmaker based on the maximum likelihood mapmaking equation:
    (P^T N^{-1} P) m = P^T N^{-1} d
or in more simple terms we solve
    A x = b
"""

import healpy as hp
import numpy as np

import globals as g


def A_dot_x(x, pointing, sigma):
    """
    Calculate the matrix-vector product A x = P^T N^{-1} P x.

    Parameters
    ----------
    x : np.ndarray
        The vector x (the map).
    pix : np.ndarray
        The pointing matrix in pixel space.
    sigma : np.ndarray
        The noise standard deviation for each time sample.

    Returns
    -------
    np.ndarray
        The result of the matrix-vector product A x.
    """

    Px = np.zeros((pointing.shape[0], g.IFG_SIZE), dtype=complex)
    for x_i in range(pointing.shape[1]):
        for pix_i, pix in enumerate(pointing[:, x_i]):
            # print(f"pointing[pix_i]: {pointing[pix_i]}")
            Px[pix_i, x_i] = x[pix, x_i]

    # check for nans
    if np.isnan(Px).any():
        raise ValueError("NaN values found in Px")

    FPx = np.fft.fft(Px, axis=1)
    N_inv_Px = FPx / (sigma[:, np.newaxis])

    # check for nans
    if np.isnan(N_inv_Px).any():
        raise ValueError("NaN values found in N_inv_Px")

    FN_inv_Px = np.fft.ifft(N_inv_Px, axis=1)

    # check for nans
    if np.isnan(FN_inv_Px).any():
        raise ValueError("NaN values found in FN_inv_Px")

    npix = hp.nside2npix(g.NSIDE)
    A_x = np.zeros((npix, g.IFG_SIZE), dtype=complex)
    for pix_i in range(pointing.shape[0]):
        A_x[pointing[pix_i]] += FN_inv_Px[pix_i]

    return A_x


def calculate_b(d, pointing, sigma):
    """
    Calculate the vector b = P^T N^{-1} d.

    Parameters
    ----------
    P_T : np.ndarray
        The hermitian conjugate of the matrix P, because we are dealing with complex numbers. In the case of simply using a Fourier transform for the P operator, this is the inverse of the Fourier transform.
    N : np.ndarray
        The matrix N.
    d : np.ndarray
        The matrix d.

    Returns
    -------
    np.ndarray
        The vector b.
    """

    Fd = np.fft.fft(d, axis=1)

    N_inv_d = Fd / (sigma[:, np.newaxis])

    FN_inv_d = np.fft.ifft(N_inv_d, axis=1)

    b = np.zeros((g.NPIX), dtype=complex)
    for pix_i in range(pointing.shape[0]):
        b[pointing[pix_i]] += FN_inv_d[pix_i]

    return b


if __name__ == "__main__":
    data = np.load("../output/ifgs_modern.npz")
    ifgs = data["ifg"]
    pix = data["pix"]
    sigma = data["sigma"].astype(int)

    print(f"pix shape: {pix.shape}")

    npix = hp.nside2npix(g.NSIDE)
    x = np.zeros((npix, g.IFG_SIZE), dtype=float)

    A_x = A_dot_x(x, pix, sigma)
    print(f"A_x: {A_x}")
