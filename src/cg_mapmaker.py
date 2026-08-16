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
def _calculate_b_numba_sigma_scalar(d, pointing, sigma_scalar, n_pix, ifg_size):
    """Numba kernel for scalar sigma."""
    inv_sigma2 = 1.0 / (sigma_scalar * sigma_scalar)

    N_inv_d = d * inv_sigma2

    b = np.zeros((n_pix, ifg_size, g.N_IFGS), dtype=np.float64)

    for ifg_i in range(g.N_IFGS):
        for x_i in range(ifg_size):
            for data_point_i in range(pointing.shape[0]):
                b[pointing[data_point_i, x_i, ifg_i], x_i, ifg_i] += N_inv_d[data_point_i, x_i, ifg_i]

    return b


@nb.njit(parallel=True, fastmath=True)
def _calculate_b_numba_sigma_vector(d, pointing, sigma, n_pix, ifg_size):
    """Numba kernel for per-sample sigma."""
    L = d.size
    d_flat = d.ravel()
    sigma_flat = sigma.ravel()
    b = np.zeros((n_pix, ifg_size, g.N_IFGS), dtype=np.float64)

    for i in nb.prange(L):
        pix = pointing[i]
        ifg = i % ifg_size
        b[pix, ifg] += d_flat[i] / (sigma_flat[i] * sigma_flat[i])

    return b.ravel()


def calculate_b_numba(d, pointing, sigma, n_pix, ifg_size):
    """
    Numba-accelerated version of calculate_b.
    Returns a flattened array of length n_pix * n_ifg.
    """
    sigma_arr = np.asarray(sigma)
    if sigma_arr.ndim == 0:
        return _calculate_b_numba_sigma_scalar(d, pointing, float(sigma_arr), n_pix, ifg_size)
    return _calculate_b_numba_sigma_vector(d, pointing, sigma_arr, n_pix, ifg_size)

# @nb.njit(parallel=True, fastmath=True)
def A_dot_x(x, pointing, sigma, n_pix):
    """
    Implementation of A @ x.
    """
    x = x.reshape((n_pix, g.IFG_SIZE[args.sim_type], g.N_IFGS))

    Pm = np.zeros((pointing.shape[0], g.IFG_SIZE[args.sim_type], g.N_IFGS), dtype=np.float64)
    for ifg_i in range(g.N_IFGS):
        for x_i in range(g.IFG_SIZE[args.sim_type]):
            for pix_i in range(n_pix):
                Pm[pointing[pix_i, x_i, ifg_i], x_i, ifg_i] = x[pix_i, x_i, ifg_i]

    # 2) Weight
    NPm = Pm / sigma**2

    # 3) Scatter back
    Ax = np.zeros_like(x)
    for ifg_i in range(g.N_IFGS):
        for x_i in range(g.IFG_SIZE[args.sim_type]):
            for data_point_i in range(pointing.shape[0]):
                pix = pointing[data_point_i, x_i, ifg_i]
                Ax[pix, x_i, ifg_i] += NPm[data_point_i, x_i, ifg_i]

    return Ax.ravel()

def preconditioned_conjugate_gradient(b, pointing, sigma, precond, x=None, maxiter=1000, tol=1e-5,
                                      npix=g.NPIX[args.sim_type], t0=None):

    t0 = utils.log_step("A_dot_x", t0, args.run_name)
    Ax = A_dot_x(x, pointing, sigma, n_pix=npix).ravel()
    b = b.ravel()

    r = b - Ax

    d = np.zeros_like(r)
    precond = precond.ravel()
    d[precond != 0] = r[precond != 0] / precond[precond != 0]

    delta_new = np.dot(r.T, d)
    delta0 = delta_new

    for i in range(maxiter):
        t0 = utils.log_step(f"PCG iteration {i+1}/{maxiter}", t0, args.run_name)
        eps = delta_new / delta0 if delta0 != 0 else 0.0
        print(f"PCG iteration {i+1}/{maxiter}, eps={eps}")
        t0 = utils.log_step(f"q A_dot_x ({i})", t0, args.run_name)
        q = A_dot_x(d, pointing, sigma, n_pix=npix).ravel()

        alpha = delta_new / np.dot(d.T, q)

        x = x.ravel() + alpha * d.ravel()

        if i % 50 == 0:
            r = b - A_dot_x(x, pointing, sigma, n_pix=npix).ravel()
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

def conjugate_gradient(b, pointing, sigma, x=None, maxiter=1000, tol=1e-5, npix=g.NPIX[args.sim_type]):
    """
    Conjugate gradient solver for Ax = b.
    """
    Ax = A_dot_x(x, pointing, sigma, n_pix=npix).ravel()
    b = b.ravel()
    r = b - Ax
    d = r.copy()
    delta_new = np.dot(r.T, r)
    delta0 = delta_new

    for i in range(maxiter):
        eps = delta_new / delta0 if delta0 != 0 else 0.0
        print(f"CG iteration {i+1}/{maxiter}, eps={eps}")
        q = A_dot_x(d, pointing, sigma, n_pix=npix).ravel()
        alpha = delta_new / np.dot(d.T, q)
        x = x.ravel() + alpha * d.ravel()

        if i % 50 == 0:
            r = b - A_dot_x(x, pointing, sigma, n_pix=npix).ravel()
        else:
            r -= alpha * q

        delta_old = delta_new
        delta_new = np.dot(r.T, r)

        beta = delta_new / delta_old
        d = r + beta * d

        if delta_new < tol**2 * delta0:
            break
    return x

@nb.njit(parallel=True, fastmath=True)
def compute_rms_map(pointing, sigma, n_pix, ifg_size):
    rms = np.zeros((n_pix, ifg_size, g.N_IFGS), dtype=np.float64)

    invsigma2 = 1.0 / sigma**2

    for ifg_i in range(g.N_IFGS):
            for x_i in range(ifg_size):
                for data_point_i in range(pointing.shape[0]):
                    pix = pointing[data_point_i, x_i, ifg_i]
                    if sigma.ndim == 0:
                        rms[pix, x_i, ifg_i] += invsigma2
                    else:
                        rms[pix, x_i, ifg_i] += invsigma2[data_point_i, x_i, ifg_i]
    
    return np.sqrt(rms)


if __name__ == "__main__":
    n_pix = g.NPIX[args.sim_type]
    ifg_size = g.IFG_SIZE[args.sim_type]

    with open(f"../output/profiling/{args.run_name}.txt", "w") as f:
        f.write("Profiling output for CG mapmaker for FOSSIL\n")
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

    if args.sim_type == "firas":
        t0 = utils.log_step("divide by n_ifgs", t0, args.run_name)
        ifgs = ifgs / g.N_IFGS

        # repeat the IFGs 16 times to match the pointing array size
        t0 = utils.log_step("repeat ifgs", t0, args.run_name)
        ifgs = np.tile(ifgs, g.N_IFGS).reshape(ecl_lon.shape)

    t0 = utils.log_step("ang2pix", t0, args.run_name)
    pix = hp.ang2pix(g.NSIDE[args.sim_type], ecl_lon, ecl_lat, lonlat=True)

    t0 = utils.log_step("calculate b_numba", t0, args.run_name)
    b = calculate_b_numba(ifgs, pix, sigma, n_pix, ifg_size)

    if args.plots == "debug":
        hp.mollview(b[:, 256, 8])
        plt.savefig("../output/debug/b_map.png")
        plt.close()

    # The CG operator uses pixel-major flattening: idx = pix * n_ifg + ifg.
    # Build x0 in 2D and ravel to avoid IFG-major ordering mistakes.
    x0_grid = np.zeros((n_pix, ifg_size), dtype=np.float64)
    for i in range(ifg_size):
        x0_grid[:, i] = hp.read_map(f"../output/noise_weighted/{args.sim_type}/ifg_maps/{i:04d}.fits")
    x0 = np.tile(x0_grid, g.N_IFGS).reshape((n_pix, ifg_size, g.N_IFGS))

    if np.isnan(x0).any():
        nan_mask = np.isnan(x0)
        x0 = np.nan_to_num(x0, nan=0.0)

    t0 = utils.log_step("preconditioned_conjugate_gradient", t0, args.run_name)
    if args.sim_type == "fossil":
        t0 = utils.log_step("compute_rms_map", t0, args.run_name)
        rms_maps = compute_rms_map(pix, sigma, n_pix, ifg_size)

        x = preconditioned_conjugate_gradient(b, pix, sigma, rms_maps,
                                            x=x0, t0=t0, npix=n_pix)
    elif args.sim_type == "firas":
        x = conjugate_gradient(b, pix, sigma, x=x0, npix=n_pix)

    x = x.reshape((n_pix, ifg_size))
    m = np.fft.rfft(x, axis=1).real
    m[nan_mask] = hp.UNSEEN

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
        f.write(f"{(_time() - t0):.2f}\n")
        f.write("=" * 50 + "\n")
        f.write(f"Total time for CG mapmaker: {(_time() - t00)/60:.2f} min\n")
