"""
Conjugate gradient mapmaker based on the maximum likelihood mapmaking equation:
    (P^T N^{-1} P) m = P^T N^{-1} d
or in more simple terms we solve
    A x = b
"""

import healpy as hp
import matplotlib.pyplot as plt
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

    Px = np.zeros(pointing.shape[0], dtype=complex)
    for pix_i, pix in enumerate(pointing):
        Px[pix_i] = x[pix]

    FPx = np.fft.fft(Px)
    N_inv_Px = FPx / sigma

    FN_inv_Px = np.fft.ifft(N_inv_Px)

    npix = hp.nside2npix(g.NSIDE)
    A_x = np.zeros(npix, dtype=complex)
    for pix_i in range(pointing.shape[0]):
        A_x[pointing[pix_i]] += FN_inv_Px[pix_i]

    return A_x


def calculate_b(d, pointing, sigma):
    """
    Calculate the vector b = P^T N^{-1} d.

    Returns
    -------
    np.ndarray
        The vector b.
    """

    Fd = np.fft.fft(d)

    N_inv_d = Fd / sigma

    FN_inv_d = np.fft.ifft(N_inv_d)

    b = np.zeros(g.NPIX, dtype=complex)
    for pix_i in range(pointing.shape[0]):
        b[pointing[pix_i]] += FN_inv_d[pix_i]

    return b


def conjugate_gradient(pointing, sigma, b, x=None, maxiter=1000, tol=1e-10):
    """
    Solve the equation A x = b using the conjugate gradient method. Taken from the Painless Conjugate Gradient paper.

    Parameters
    ----------
    pointing : np.ndarray
        The pointing matrix in pixel space.
    sigma : np.ndarray
        The noise standard deviation for each time sample.
    b : np.ndarray
        The vector b.
    x : np.ndarray, optional
        The initial guess for the solution x. If None, a zero vector is used.
    maxiter : int, optional
        The maximum number of iterations. Default is 1000.
    tol : float, optional
        The tolerance for convergence. Default is 1e-10.

    Returns
    -------
    np.ndarray
        The solution vector x.
    """
    if x is None:
        x = np.zeros_like(b)

    Ax = A_dot_x(x, pointing, sigma)
    r = b - Ax

    d = r

    delta_new = np.dot(r.T, r)
    delta0 = delta_new

    for i in range(maxiter):
        q = A_dot_x(d, pointing, sigma)

        alpha = delta_new / np.dot(d.T, q)

        x += alpha * d

        if i % 50 == 0:
            r = b - A_dot_x(x, pointing, sigma)
        else:
            r -= alpha * q

        delta_old = delta_new
        delta_new = np.dot(r.T, r)

        beta = delta_new / delta_old
        d = r + beta * d

        if delta_new < tol**2 * delta0:
            break

    return x


if __name__ == "__main__":
    data = np.load(f"../output/ifgs_{g.SIM_TYPE}.npz")
    ifgs = data["ifg"]
    pix = data["pix"]
    sigma = data["sigma"]

    if g.SIM_TYPE == "firas":
        ifgs = ifgs / g.N_IFGS
    ifgs = np.roll(ifgs, -360, axis=1)

    x = np.zeros((g.NPIX, g.IFG_SIZE), dtype=complex)
    for freq_i in range(ifgs.shape[1]):
        print(f"Solving for frequency index {freq_i+1}/{ifgs.shape[1]}")
        b = calculate_b(ifgs[:, freq_i], pix[:, freq_i], sigma)
        x[:, freq_i] = conjugate_gradient(
            pix[:, freq_i], sigma, b, maxiter=10, tol=1e-5
        )

    m = np.abs(np.fft.rfft(x, axis=1))
    print("Finished CG mapmaking, saving to disk...")

    for nui in range(m.shape[1]):
        if g.FITS:
            hp.write_map(
                f"./../output/cg_mapmaker/{g.SIM_TYPE}/{int(nui):04d}.fits",
                m[:, nui],
                overwrite=True,
                dtype=np.float64,
            )
        if g.PNG:
            hp.mollview(
                m[:, nui],
                title=f"{int(nui):04d} GHz",
                unit="MJy/sr",
                min=0,
                max=50,
                xsize=2000,
                coord=["E", "G"],
            )
            plt.savefig(f"../output/cg_mapmaker/{g.SIM_TYPE}/{int(nui):04d}.png")
            plt.close()
