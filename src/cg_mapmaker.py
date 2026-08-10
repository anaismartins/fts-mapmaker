"""
Conjugate gradient mapmaker based on the maximum likelihood mapmaking equation:
    (P^T N^{-1} P) m = P^T N^{-1} d
or in more simple terms we solve
    A x = b
"""

from time import time as _time

import healpy as hp
import matplotlib.pyplot as plt
import numba as nb
import numpy as np

import globals as g
import spectra
import utils
from argparser import args


@nb.njit(parallel=True, fastmath=True)
def _calculate_b_numba_sigma_scalar(d, pointing, sigma_scalar, n_pix, n_ifg):
    """Numba kernel for scalar sigma."""
    L = d.size
    d_flat = d.ravel()
    inv_sigma2 = 1.0 / (sigma_scalar * sigma_scalar)
    b = np.zeros((n_pix, n_ifg), dtype=np.float64)

    for i in nb.prange(L):
        pix = pointing[i]
        ifg = i % n_ifg
        b[pix, ifg] += d_flat[i] * inv_sigma2

    return b.ravel()


@nb.njit(parallel=True, fastmath=True)
def _calculate_b_numba_sigma_vector(d, pointing, sigma, n_pix, n_ifg):
    """Numba kernel for per-sample sigma."""
    L = d.size
    d_flat = d.ravel()
    sigma_flat = sigma.ravel()
    b = np.zeros((n_pix, n_ifg), dtype=np.float64)

    for i in nb.prange(L):
        pix = pointing[i]
        ifg = i % n_ifg
        b[pix, ifg] += d_flat[i] / (sigma_flat[i] * sigma_flat[i])

    return b.ravel()


def calculate_b_numba(d, pointing, sigma, n_pix, n_ifg):
    """
    Numba-accelerated version of calculate_b.
    Returns a flattened array of length n_pix * n_ifg.
    """
    sigma_arr = np.asarray(sigma)
    if sigma_arr.ndim == 0:
        return _calculate_b_numba_sigma_scalar(d, pointing, float(sigma_arr), n_pix, n_ifg)
    return _calculate_b_numba_sigma_vector(d, pointing, sigma_arr, n_pix, n_ifg)

def A_dot_x_vectorised(x, pointing, sigma, n_pix, n_ifg, t0):
    """
    Vectorised implementation of A @ x.
    """
    t0 = utils.log_step("A_dot_x_vectorised", t0, args.run_name)
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

    Ax = A_dot_x_vectorised(x, pointing, sigma, n_pix=npix, n_ifg=n_ifgs, t0=t0)

    r = b - Ax

    d = np.zeros_like(r)
    d[precond != 0] = r[precond != 0] / precond[precond != 0]

    delta_new = np.dot(r.T, d)
    delta0 = delta_new

    for i in range(maxiter):
        t0 = utils.log_step(f"PCG iteration {i+1}/{maxiter}", t0, args.run_name)
        eps = delta_new / delta0 if delta0 != 0 else 0.0
        print(f"PCG iteration {i+1}/{maxiter}, eps={eps}")
        t0 = utils.log_step(f"q A_dot_x_vectorised ({i})", t0, args.run_name)
        q = A_dot_x_vectorised(d, pointing, sigma, n_pix=npix, n_ifg=n_ifgs, t0=t0)

        alpha = delta_new / np.dot(d.T, q)

        x += alpha * d

        if i % 50 == 0:
            r = b - A_dot_x_vectorised(x, pointing, sigma, n_pix=npix, n_ifg=n_ifgs, t0=t0)
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

def compute_rms_map(pointing, sigma, n_pix, n_ifg, t0):
    t0 = utils.log_step("compute_rms_map", t0, args.run_name)
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

    if args.sim_type == "fossil":
        t0 = utils.log_step("roll", t0, args.run_name)
        ifgs = np.roll(ifgs, -180, axis=1)
    elif args.sim_type == "firas":
        t0 = utils.log_step("divide by n_ifgs", t0, args.run_name)
        ifgs = ifgs / g.N_IFGS
        t0 = utils.log_step("roll", t0, args.run_name)
        ifgs = np.roll(ifgs, -360, axis=1)

    t0 = utils.log_step("ang2pix", t0, args.run_name)
    pix = hp.ang2pix(g.NSIDE[args.sim_type], ecl_lon, ecl_lat, lonlat=True).flatten()

    t0 = utils.log_step("calculate b_numba", t0, args.run_name)
    b = calculate_b_numba(ifgs, pix, sigma, n_pix, n_ifgs)

    t0 = utils.log_step("compute hits map vectorised", t0, args.run_name)
    hit_maps = compute_hits_map(pix, n_pix, n_ifgs)

    rms_maps = compute_rms_map(pix, sigma, n_pix, n_ifgs, t0)

    # The CG operator uses pixel-major flattening: idx = pix * n_ifg + ifg.
    # Build x0 in 2D and ravel to avoid IFG-major ordering mistakes.
    x0_grid = np.zeros((n_pix, n_ifgs), dtype=np.float64)
    for i in range(n_ifgs):
        x0_grid[:, i] = hp.read_map(f"../output/white_noise/{args.sim_type}/ifg_maps/{i:04d}.fits")
    x0 = x0_grid.ravel()

    t0 = utils.log_step("preconditioned_conjugate_gradient", t0, args.run_name)
    x = preconditioned_conjugate_gradient(b, pix, sigma, rms_maps, x=x0, t0=t0)

    x = x.reshape((n_pix, n_ifgs))
    m = np.real(np.fft.rfft(x, axis=1))

    # use the solution of the white noise mapmaker as x0
    if args.sim_type == "fossil":
        nfreq = 129
    elif args.sim_type == "firas":
        nfreq = 257
    else:
        raise ValueError("Unknown sim_type")
    frequencies = spectra.generate_frequencies(simtype=args.sim_type, nfreq=nfreq)

    path = f"../output/cg/{args.sim_type}/"
    for nui, freq in enumerate(frequencies):
        if g.FITS:
            hp.write_map(f"{path}{int(freq):04d}.fits", m[:, nui], overwrite=True, dtype=np.float64)
        if g.PNG:
            hp.mollview(m[:, nui], title=f"{int(freq):04d} GHz", unit="MJy/sr", min=0, max=50,
                        coord=["E", "G"])
            plt.savefig(f"{path}{int(freq):04d}.png")
            plt.close()

    t0 = utils.log_step("save_maps", t0, args.run_name)
    print(f"Saved maps to ../output/cg/{args.sim_type}/.")
        
    with open(f"../output/profiling/{args.run_name}.txt", "a") as f:
        f.write("=" * 50 + "\n")
        f.write(f"Total time for binned mapmaker: {(_time() - t00)/60:.2f} min\n")
