"""
Conjugate gradient mapmaker based on the maximum likelihood mapmaking equation:
    (P^T N^{-1} P) m = P^T N^{-1} d
or in more simple terms we solve
    A x = b
"""

import multiprocessing
import time

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np

import globals as g
import utils


def calculate_b(d, pointing, sigma):
    """
    Calculate the vector b = P^T N^{-1} d.

    Returns
    -------
    np.ndarray
        The vector b.
    """

    # Fd = np.fft.rfft(d, axis=1).flatten()

    # N_inv_d = (Fd / sigma**2).reshape((d.shape[0], g.SPEC_SIZE))

    # FN_inv_d = np.fft.irfft(N_inv_d, axis=1).flatten()

    N_inv_d = d.flatten() / sigma**2
    # N_inv_d = np.fft.fft(d) / sigma**2

    # N_inv_d = np.fft.ifft(N_inv_d)

    b = np.zeros(
        (
            g.NPIX,
            g.IFG_SIZE,
        ),
        dtype=complex,
    )
    for pix_i in range(d.shape[0]):
        for x_i in range(d.shape[1]):
            b[pointing[pix_i * g.IFG_SIZE + x_i], x_i] += N_inv_d[
                pix_i * g.IFG_SIZE + x_i
            ]

    return b.flatten()


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

    x = x.reshape((g.NPIX, g.IFG_SIZE))

    Px = np.zeros((pointing.shape[0] // g.IFG_SIZE, g.IFG_SIZE), dtype=complex)
    for pix_i in range(pointing.shape[0] // g.IFG_SIZE):
        for x_i in range(g.IFG_SIZE):
            Px[pix_i, x_i] = x[pointing[pix_i * g.IFG_SIZE + x_i], x_i]

    # FPx = np.fft.rfft(Px, axis=1).flatten()
    # N_inv_Px = (FPx / sigma**2).reshape((Px.shape[0], g.SPEC_SIZE))
    N_inv_Px = Px.flatten() / sigma**2

    # FN_inv_Px = np.fft.irfft(N_inv_Px, axis=1).flatten()

    A_x = np.zeros((g.NPIX, g.IFG_SIZE), dtype=complex)
    for pix_i in range(pointing.shape[0] // g.IFG_SIZE):
        for x_i in range(g.IFG_SIZE):
            A_x[pointing[pix_i * g.IFG_SIZE + x_i], x_i] += N_inv_Px[
                pix_i * g.IFG_SIZE + x_i
            ]

    return A_x.flatten()


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
        print(f"CG iteration {i+1}/{maxiter}, eps={delta_new/delta0}")
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

        y = x.reshape((g.NPIX, g.IFG_SIZE))
        m = np.abs(np.fft.rfft(y, axis=1))

        r2 = r.reshape((g.NPIX, g.IFG_SIZE))

        hp.mollview(
            m[:, 100],
            title="CG map at frequency index 100",
            unit="Amplitude",
            min=0,
            max=20,
            coord=["E", "G"],
        )
        plt.savefig(f'../output/cg_iter_{i:04}.png')

        hp.mollview(
            y[:, 100],
            title="IFG map at distance index 100",
            unit="Amplitude",
            coord=["E", "G"],
        )
        plt.savefig(f'../output/cg_ifg_iter_{i:04}.png')
        hp.mollview(
            r2[:, 100],
            title="IFG map at distance index 100",
            unit="Amplitude",
            coord=["E", "G"],
        )
        plt.savefig(f'../output/cg_res_ifg_iter_{i:04}.png')
        plt.close()

    return x


def preconditioned_conjugate_gradient(
    b, pointing, sigma, hits_map, x=None, maxiter=1000, tol=1e-10
):
    if x is None:
        x = np.zeros_like(b)

    Ax = A_dot_x(x, pointing, sigma)
    r = b - Ax

    # d = M_inv @ r
    d = np.zeros_like(r)
    d[hits_map != 0] = r[hits_map != 0] / hits_map[hits_map != 0]
    delta_new = np.dot(r.T, d)
    delta0 = delta_new

    for i in range(maxiter):
        print(f"PCG iteration {i+1}/{maxiter}")
        q = A_dot_x(d, pointing, sigma)

        alpha = delta_new / np.dot(d.T, q)

        x += alpha * d

        if i % 50 == 0:
            r = b - A_dot_x(x, pointing, sigma)
        else:
            r -= alpha * q

        # s = M_inv @ r
        s = np.zeros_like(r)
        s[hits_map != 0] = r[hits_map != 0] / hits_map[hits_map != 0]
        delta_old = delta_new
        delta_new = np.dot(r.T, s)

        beta = delta_new / delta_old
        d = s + beta * d

        if delta_new < tol**2 * delta0:
            break


# def solve_frequency(freq_i, ifg_data, pix_data, sigma):
#     """Solve for a single frequency index."""
#     print(f"{(freq_i+1):03d}/{ifgs.shape[1]}: Started.")
#     t1 = time.time()

#     b = calculate_b(ifg_data, pix_data, sigma)
#     x_freq = conjugate_gradient(pix_data, sigma, b, tol=1e-4, freq_i=freq_i)

#     t2 = time.time()
#     print(f"{(freq_i+1):03d}/{ifgs.shape[1]}: Finished in {int((t2 - t1))} seconds.")

#     return freq_i, x_freq


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
    print(f"shape of ifgs before flatten: {ifgs.shape}")

    # ifgs = ifgs.flatten()
    # print(f"shape of ifgs after flatten: {ifgs.shape}")

    print(f"shape of pix before flatten: {pix.shape}")
    pix = pix.flatten()
    print(f"shape of pix after flatten: {pix.shape}")

    print(f"shape of sigma before flatten: {sigma.shape}")
    sigma = (sigma[:, np.newaxis] * np.ones(g.IFG_SIZE)).flatten()
    print(f"shape of sigma after flatten: {sigma.shape}")

    plt.vlines(
        512 * np.arange(10),
        np.min(sigma[: (512 * 10)]),
        np.max(sigma[: (512 * 10)]),
        color="red",
    )
    plt.vlines(
        360 + 512 * np.arange(10),
        np.min(sigma[: (512 * 10)]),
        np.max(sigma[: (512 * 10)]),
        color="green",
    )
    plt.plot(sigma[: (512 * 10)])
    # plt.plot(ifgs[: (512 * 10)], alpha=0.5, color="orange")
    plt.plot(pix[: (512 * 10)], alpha=0.5, color="purple")
    # plt.show()
    plt.savefig("../output/ifgs_debug.png")
    plt.close()

    b = calculate_b(ifgs, pix, sigma)

    # debug
    b_map = b.reshape((g.NPIX, g.IFG_SIZE))
    b_map = np.abs(np.fft.rfft(b_map, axis=1))
    # hp.mollview(
    #     b_map[:, 100],
    #     title="b map at frequency index 100",
    #     unit="Amplitude",
    #     min=0,
    #     max=200,
    #     coord=["E", "G"],
    # )
    # plt.show()
    # plt.savefig("../output/b_map_debug.png")
    # plt.close()

    print(f"Starting conjugate gradient solver...")

    # set M to be the hits map
    hits_map = np.zeros((g.NPIX, g.IFG_SIZE))
    for pix_i in range(pix.shape[0] // g.IFG_SIZE):
        for x_i in range(g.IFG_SIZE):
            hits_map[pix[pix_i * g.IFG_SIZE + x_i], x_i] += 1
    # M_inv = np.diag(1 / hits_map.flatten())
    hits_map = hits_map.flatten()
    x = preconditioned_conjugate_gradient(b, pix, sigma, hits_map)

    x = x.reshape((g.NPIX, g.IFG_SIZE))
    m = np.abs(np.fft.rfft(x, axis=1))
    t2 = time.time()
    print(f"Finished CG mapmaking in {int((t2 - t1)/60)} minutes.")
    print("Finished CG mapmaking, saving to disk...")

    frequencies = utils.generate_frequencies("ll", "ss", 257)
    t1 = time.time()
    # Save maps serially to avoid threading issues with matplotlib/Qt
    for nui, freq in enumerate(frequencies):
        utils.save_maps(freq, m[:, nui])
    t2 = time.time()
    print(f"Finished saving maps in {int((t2 - t1))} seconds.")
