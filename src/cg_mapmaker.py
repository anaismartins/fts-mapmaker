"""
Conjugate gradient mapmaker based on the maximum likelihood mapmaking equation:
    (P^T N^{-1} P) m = P^T N^{-1} d
or in more simple terms we solve
    A x = b
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from time import time as _time

import healpy as hp
import numba as nb
import numpy as np

import globals as g
import spectra
import utils
from argparser import args


def calculate_b(d, pointing, sigma, t0):
    """
    Calculate the vector b = P^T N^{-1} d.

    Returns
    -------
    np.ndarray
        The vector b.
    """
    n_ifgs = g.IFG_SIZE[args.sim_type]

    t0 = utils.log_step("calculate N_inv_d", _time(), args.run_name)
    N_inv_d = d.flatten() / sigma**2

    t0 = utils.log_step("initialize b", t0, args.run_name)
    b = np.zeros((g.NPIX[args.sim_type], n_ifgs), dtype=np.float64)
    t0 = utils.log_step("compute b", t0, args.run_name)
    for pix_i in range(d.shape[0]):
        for x_i in range(d.shape[1]):
            b[pointing[pix_i * n_ifgs + x_i], x_i] += N_inv_d[pix_i * n_ifgs + x_i]

    return b.flatten(), t0

@nb.njit(parallel=True, fastmath=True)
def calculate_b_numba(d, pointing, sigma, n_pix, n_ifg):
    """
    Numba‑accelerated version of calculate_b.
    Returns a flattened array of length n_pix * n_ifg.
    """
    L = d.size
    b = np.zeros((n_pix, n_ifg), dtype=np.float64)

    for i in nb.prange(L):
        pix = pointing[i]
        ifg = i % n_ifg
        b[pix, ifg] += d.ravel()[i] / sigma[i]**2

    return b.ravel()



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
    n_ifgs = g.IFG_SIZE[args.sim_type]

    x = x.reshape((npix, n_ifgs))

    Px = np.zeros((pointing.shape[0] // n_ifgs, n_ifgs),
                  dtype=np.float64)
    for pix_i in range(pointing.shape[0] // n_ifgs):
        for x_i in range(n_ifgs):
            Px[pix_i, x_i] = x[pointing[pix_i * n_ifgs + x_i], x_i]

    N_inv_Px = Px.flatten() / sigma**2

    A_x = np.zeros((npix, n_ifgs), dtype=np.float64)
    for pix_i in range(pointing.shape[0] // n_ifgs):
        for x_i in range(n_ifgs):
            A_x[pointing[pix_i * n_ifgs + x_i], x_i] += N_inv_Px[pix_i * n_ifgs + x_i]

    return A_x.flatten()

def A_dot_x_vectorised(x, pointing, sigma, n_pix, n_ifg):
    """
    Vectorised implementation of A @ x.
    """
    x_grid = x.reshape((n_pix, n_ifg))

    # 1) Gather: map each sample to the corresponding x entry
    flat_x = x_grid[pointing, np.arange(pointing.size) % n_ifg]

    # 2) Weight
    weighted = flat_x / sigma**2

    # 3) Scatter back
    Ax = np.zeros_like(x_grid)
    np.add.at(Ax, (pointing, np.arange(pointing.size) % n_ifg), weighted)

    return Ax.ravel()

def preconditioned_conjugate_gradient(b, pointing, sigma, precond, x=None, maxiter=1000, tol=1e-10,
                                      npix=g.NPIX[args.sim_type], t0=None):
    n_ifgs = g.IFG_SIZE[args.sim_type]
    if x is None:
        x = np.zeros_like(b)
    else:
        x = np.asarray(x, dtype=np.float64)

    b = np.asarray(b, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    precond = np.asarray(precond, dtype=np.float64)

    t0 = utils.log_step("A_dot_x", t0, args.run_name)
    Ax = A_dot_x(x, pointing, sigma, npix=npix)
    t0 = utils.log_step("A_dot_x_vectorised", t0, args.run_name)
    _Ax = A_dot_x_vectorised(x, pointing, sigma, npix=npix, n_ifg=n_ifgs)
    if not np.allclose(Ax, _Ax):
        raise ValueError("A_dot_x and A_dot_x_vectorised are not equal!")
    else:
        print("A_dot_x and A_dot_x_vectorised are equal!")
    r = b - Ax

    d = np.zeros_like(r)
    d[precond != 0] = r[precond != 0] / precond[precond != 0]

    delta_new = np.dot(r.T, d)
    delta0 = delta_new

    for i in range(maxiter):
        t0 = utils.log_step(f"PCG iteration {i+1}/{maxiter}", t0, args.run_name)
        eps = delta_new / delta0 if delta0 != 0 else 0.0
        print(f"PCG iteration {i+1}/{maxiter}, eps={eps}")
        t0 = utils.log_step(f"q A_dot_x ({i})", t0, args.run_name)
        q = A_dot_x(d, pointing, sigma, npix=npix)
        t0 = utils.log_step(f"q A_dot_x_vectorised ({i})", t0, args.run_name)
        _q = A_dot_x_vectorised(d, pointing, sigma, npix=npix, n_ifg=n_ifgs)
        if not np.allclose(q, _q):
            raise ValueError("q and _q are not equal!")
        else:
            print("q and _q are equal!")

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

def compute_hits_map(pointing, n_pix, n_ifg):
    hits = np.zeros((n_pix, n_ifg), dtype=np.int64)
    freq_idx = np.arange(pointing.size) % n_ifg
    np.add.at(hits, (pointing, freq_idx), 1)
    return hits.ravel()

def compute_rms_map(pointing, sigma, n_pix, n_ifg):
    rms = np.zeros((n_pix, n_ifg), dtype=np.float64)
    freq_idx = np.arange(pointing.size) % n_ifg
    invsigma2 = 1.0 / sigma**2
    np.add.at(rms, (pointing, freq_idx), invsigma2)
    return np.sqrt(rms.ravel())


if __name__ == "__main__":
    n_ifgs = g.IFG_SIZE[args.sim_type]
    n_pix = g.NPIX[args.sim_type]

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

    b, t0 = calculate_b(ifgs, pix, sigma, t0)
    t0 = utils.log_step("calculate b_numba", t0, args.run_name)
    _b = calculate_b_numba(ifgs, pix, sigma, n_pix, n_ifgs)

    # compare b and _b
    if not np.allclose(b, _b):
        raise ValueError("b and _b are not equal!")
    else:
        print("b and _b are equal!")

    # set M to be the hits map
    t0 = utils.log_step("compute hits map", t0, args.run_name)
    hits_map = np.zeros((n_pix, n_ifgs))
    for pix_i in range(pix.shape[0] // n_ifgs):
        for x_i in range(n_ifgs):
            hits_map[pix[pix_i * n_ifgs + x_i], x_i] += 1
    hits_map = hits_map.flatten()

    t0 = utils.log_step("compute hits map vectorised", t0, args.run_name)
    _hit_maps = compute_hits_map(pix, n_pix, n_ifgs)
    if not np.allclose(hits_map, _hit_maps):
        raise ValueError("hits_map and _hit_maps are not equal!")
    else:
        print("hits_map and _hit_maps are equal!")

    t0 = utils.log_step("compute rms map", t0, args.run_name)
    rms_map = np.zeros((n_pix, n_ifgs))
    for pix_i in range(pix.shape[0] // n_ifgs):
        for x_i in range(n_ifgs):
            rms_map[pix[pix_i * n_ifgs + x_i], x_i] += (1 / sigma ** 2)
    rms_map = np.sqrt(rms_map.flatten())
    t0 = utils.log_step("compute rms map vectorised", t0, args.run_name)
    _rms_maps = compute_rms_map(pix, sigma, n_pix, n_ifgs)
    if not np.allclose(rms_map, _rms_maps):
        raise ValueError("rms_map and _rms_maps are not equal!")
    else:
        print("rms_map and _rms_maps are equal!")

    x0 = np.zeros_like(b)
    for i in range(n_ifgs):
        x0[n_pix * i : n_pix * (i + 1)] = hp.read_map(
            f"../output/white_noise/{args.sim_type}/ifg_maps/{i:04d}.fits")

    t0 = utils.log_step("preconditioned_conjugate_gradient", t0, args.run_name)
    x = preconditioned_conjugate_gradient(b, pix, sigma, rms_map, x=x0, t0=t0)

    x = x.reshape((n_pix, n_ifgs))
    m = np.real(np.fft.rfft(x, axis=1))

    # use the solution of the white noise mapmaker as x0
    if args.sim_type == "fossil":
        nfreq = 129
    elif args.sim_type == "firas":
        nfreq = 257
    else:
        raise ValueError("Unknown sim_type")
    frequencies = spectra.generate_frequencies(nfreq=nfreq)

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
