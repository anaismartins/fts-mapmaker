"""
Conjugate gradient mapmaker based on the maximum likelihood mapmaking equation:
    (P^T N^{-1} P) m = P^T N^{-1} d
or in more simple terms we solve
    A x = b
"""

import multiprocessing
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np

import globals as g
import utils


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
    N_inv_Px = FPx / sigma  # **2

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

    N_inv_d = Fd / sigma  # **2

    FN_inv_d = np.fft.ifft(N_inv_d)

    b = np.zeros(g.NPIX, dtype=complex)
    for pix_i in range(pointing.shape[0]):
        b[pointing[pix_i]] += FN_inv_d[pix_i]

    return b


def conjugate_gradient(
    pointing, sigma, b, x=None, maxiter=1000, tol=1e-10, freq_i=None
):
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
            print(
                f"{(freq_i+1):03d}/{ifgs.shape[1]}: Conjugate gradient converged in {i+1} iterations."
            )
            break

    return x


def solve_frequency(freq_i, ifg_data, pix_data, sigma):
    """Solve for a single frequency index."""
    print(f"{(freq_i+1):03d}/{ifgs.shape[1]}: Started.")
    t1 = time.time()

    b = calculate_b(ifg_data, pix_data, sigma)
    x_freq = conjugate_gradient(pix_data, sigma, b, tol=1e-6, freq_i=freq_i)

    t2 = time.time()
    print(f"{(freq_i+1):03d}/{ifgs.shape[1]}: Finished in {int((t2 - t1))} seconds.")

    return freq_i, x_freq


if __name__ == "__main__":
    t1 = time.time()
    print("Initializing CG mapmaker...")
    data = np.load(f"../output/ifgs_{g.SIM_TYPE}.npz")
    ifgs = data["ifg"]
    pix = data["pix"]
    sigma = data["sigma"]

    max_threads = multiprocessing.cpu_count()
    print(f"Number of threads: {max_threads}")

    # Use ThreadPoolExecutor for parallel processing
    max_workers = min(ifgs.shape[1], max_threads)  # Adjust number of threads as needed

    if g.SIM_TYPE == "firas":
        ifgs = ifgs / g.N_IFGS
    ifgs = np.roll(ifgs, -360, axis=1)

    x = np.zeros((g.NPIX, g.IFG_SIZE), dtype=complex)
    # for freq_i in range(ifgs.shape[1]):
    #     t1 = time.time()
    #     print(f"Solving for frequency index {freq_i+1}/{ifgs.shape[1]}")
    #     b = calculate_b(ifgs[:, freq_i], pix[:, freq_i], sigma)
    #     x[:, freq_i] = conjugate_gradient(pix[:, freq_i], sigma, b, tol=1e-4)

    #     # Ma = linalg.LinearOperator(
    #     #     (g.NPIX, g.NPIX),
    #     #     matvec=lambda x: A_dot_x(x, pix[:, freq_i], sigma),
    #     # )
    #     # x[:, freq_i], info = linalg.cg(Ma, b)
    #     # if info != 0:
    #     #     print(f"Conjugate gradient did not converge.")

    #     t2 = time.time()
    #     print(
    #         f"Time taken for frequency index {freq_i+1}: {int((t2 - t1))} seconds. Estimated time left: {(t2 - t1) * (ifgs.shape[1] - freq_i - 1)/60:.2f} minutes."
    #     )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = {
            executor.submit(
                solve_frequency, freq_i, ifgs[:, freq_i], pix[:, freq_i], sigma
            ): freq_i
            for freq_i in range(ifgs.shape[1])
        }

        # Collect results as they complete
        for future in as_completed(futures):
            freq_i, x_freq = future.result()
            x[:, freq_i] = x_freq

    m = np.abs(np.fft.rfft(x, axis=1))
    t2 = time.time()
    print(f"Finished CG mapmaking in {int((t2 - t1)/60)} minutes.")
    print("Finished CG mapmaking, saving to disk...")

    frequencies = utils.generate_frequencies("ll", "ss", 257)
    t1 = time.time()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for nui, freq in enumerate(frequencies):
            futures.append(executor.submit(utils.save_maps, freq, m[:, nui]))
        # Ensure all are completed
        for future in as_completed(futures):
            future.result()
    t2 = time.time()
    print(f"Finished saving maps in {int((t2 - t1))} seconds.")
