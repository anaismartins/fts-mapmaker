"""
Conjugate gradient mapmaker based on the maximum likelihood mapmaking equation:
    (P^T N^{-1} P) m = P^T N^{-1} d
or in more simple terms we solve
    A x = b
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from time import time as _time

import healpy as hp
import numpy as np

import globals as g
import utils
from argparser import args


def calculate_b(d, pointing, sigma):
    """
    Calculate the vector b = P^T N^{-1} d.

    Returns
    -------
    np.ndarray
        The vector b.
    """

    N_inv_d = d.flatten() / sigma**2

    b = np.zeros((g.NPIX[args.sim_type], g.IFG_SIZE[args.sim_type],), dtype=np.float64)
    for pix_i in range(d.shape[0]):
        for x_i in range(d.shape[1]):
            b[pointing[pix_i * g.IFG_SIZE[args.sim_type] + x_i], x_i] += N_inv_d[pix_i *
                                                                                 g.IFG_SIZE[
                                                                                     args.sim_type
                                                                                     ] + x_i]

    return b.flatten()


def A_dot_x(x, pointing, sigma, npix=g.NPIX[args.sim_type]):
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

    x = x.reshape((npix, g.IFG_SIZE[args.sim_type]))

    Px = np.zeros((pointing.shape[0] // g.IFG_SIZE[args.sim_type], g.IFG_SIZE[args.sim_type]),
                  dtype=np.float64)
    for pix_i in range(pointing.shape[0] // g.IFG_SIZE[args.sim_type]):
        for x_i in range(g.IFG_SIZE[args.sim_type]):
            Px[pix_i, x_i] = x[pointing[pix_i * g.IFG_SIZE[args.sim_type] + x_i], x_i]

    N_inv_Px = Px.flatten() / sigma**2

    A_x = np.zeros((npix, g.IFG_SIZE[args.sim_type]), dtype=np.float64)
    for pix_i in range(pointing.shape[0] // g.IFG_SIZE[args.sim_type]):
        for x_i in range(g.IFG_SIZE[args.sim_type]):
            A_x[pointing[pix_i * g.IFG_SIZE[args.sim_type] + x_i], x_i] += N_inv_Px[
                pix_i *g.IFG_SIZE[args.sim_type] + x_i]

    return A_x.flatten()


def preconditioned_conjugate_gradient(b, pointing, sigma, precond, x=None, maxiter=1000, tol=1e-10,
                                      npix=g.NPIX[args.sim_type], t0=None):
    if x is None:
        x = np.zeros_like(b)
    else:
        x = np.asarray(x, dtype=np.float64)

    b = np.asarray(b, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    precond = np.asarray(precond, dtype=np.float64)

    Ax = A_dot_x(x, pointing, sigma, npix=npix)
    print(f"b: {b}")
    r = b - Ax

    d = np.zeros_like(r)
    d[precond != 0] = r[precond != 0] / precond[precond != 0]

    delta_new = np.dot(r.T, d)
    delta0 = delta_new

    for i in range(maxiter):
        t0 = utils.log_step(f"PCG iteration {i+1}/{maxiter}", t0, args.run_name)
        eps = delta_new / delta0 if delta0 != 0 else 0.0
        print(f"PCG iteration {i+1}/{maxiter}, eps={eps}")
        q = A_dot_x(d, pointing, sigma, npix=npix)

        alpha = delta_new / np.dot(d.T, q)

        x += alpha * d

        if i % 50 == 0:
            r = b - A_dot_x(x, pointing, sigma, npix=npix)
        else:
            r -= alpha * q

        # s = M_inv @ r
        s = np.zeros_like(r)
        s[precond != 0] = r[precond != 0] / precond[precond != 0]
        delta_old = delta_new
        delta_new = np.dot(r.T, s)

        beta = delta_new / delta_old
        d = s + beta * d

        if delta_new < tol**2 * delta0:
            break
    return x


if __name__ == "__main__":
    with open(f"../output/profiling/{args.run_name}.txt", "w") as f:
        f.write("Profiling output for binned mapmaker for FOSSIL\n")
        f.write("=" * 50 + "\n")
        f.write(f"{'starting':<40} | ")

    t00 = _time()
    t0 = _time()

    t0 = utils.log_step("load ifgs", t0, args.run_name)
    ifgs = np.load(f"../output/data/{args.sim_type}/ifgs.npy", mmap_mode="r")
    t0 = utils.log_step("load pix", t0, args.run_name)
    ecl_lon = np.load(f"../output/data/{args.sim_type}/ecl_lon.npy", mmap_mode="r")
    ecl_lat = np.load(f"../output/data/{args.sim_type}/ecl_lat.npy", mmap_mode="r")
    t0 = utils.log_step("load sigma", t0, args.run_name)
    sigma = np.load(f"../output/data/{args.sim_type}/noise.npy", mmap_mode="r")

    t0 = utils.log_step("roll", t0, args.run_name)
    if args.sim_type == "fossil":
        ifgs = np.roll(ifgs, -180, axis=1)
    elif args.sim_type == "firas":
        ifgs = ifgs / g.N_IFGS
        ifgs = np.roll(ifgs, -360, axis=1)

    t0 = utils.log_step("ang2pix", t0, args.run_name)
    pix = hp.ang2pix(g.NSIDE[args.sim_type], ecl_lon, ecl_lat, lonlat=True).flatten()

    t0 = utils.log_step("compute b", t0, args.run_name)
    b = calculate_b(ifgs, pix, sigma)

    # set M to be the hits map
    hits_map = np.zeros((g.NPIX[args.sim_type], g.IFG_SIZE[args.sim_type]))
    for pix_i in range(pix.shape[0] // g.IFG_SIZE[args.sim_type]):
        for x_i in range(g.IFG_SIZE[args.sim_type]):
            hits_map[pix[pix_i * g.IFG_SIZE[args.sim_type] + x_i], x_i] += 1
    hits_map = hits_map.flatten()

    rms_map = np.zeros((g.NPIX[args.sim_type], g.IFG_SIZE[args.sim_type]))
    for pix_i in range(pix.shape[0] // g.IFG_SIZE[args.sim_type]):
        for x_i in range(g.IFG_SIZE[args.sim_type]):
            rms_map[pix[pix_i * g.IFG_SIZE[args.sim_type] + x_i], x_i] += (1 / sigma ** 2)
    rms_map = np.sqrt(rms_map.flatten())

    x0 = np.zeros_like(b)
    for i in range(g.IFG_SIZE[args.sim_type]):
        x0[g.NPIX[args.sim_type] * i : g.NPIX[args.sim_type] * (i + 1)] = hp.read_map(
            f"../output/white_noise/{args.sim_type}/ifg_maps/{i:04d}.fits")

    t0 = utils.log_step("preconditioned_conjugate_gradient", t0, args.run_name)
    x = preconditioned_conjugate_gradient(b, pix, sigma, rms_map, x=x0, t0=t0)

    x = x.reshape((g.NPIX[args.sim_type], g.IFG_SIZE[args.sim_type]))
    m = np.real(np.fft.rfft(x, axis=1))

    # use the solution of the white noise mapmaker as x0
    if args.sim_type == "fossil":
        nfreq = 129
    elif args.sim_type == "firas":
        nfreq = 257
    else:
        raise ValueError("Unknown sim_type")
    frequencies = utils.generate_frequencies(nfreq=nfreq)

    with ThreadPoolExecutor(max_workers=args.nworkers) as executor:
        futures = []
        for nui, freq in enumerate(frequencies):
            futures.append(executor.submit(utils.save_maps, freq, m[:, nui]))
        # Ensure all are completed
        for future in as_completed(futures):
            future.result()

    t0 = utils.log_step("save_maps", t0, args.run_name)
    print(f"Saved maps to ../output/binned/{args.sim_type}/.")
        
    with open(f"../output/profiling/{args.run_name}.txt", "a") as f:
        f.write("=" * 50 + "\n")
        f.write(f"Total time for binned mapmaker: {(_time() - t00)/60:.2f} min\n")
