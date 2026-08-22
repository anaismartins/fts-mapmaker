"""
Conjugate gradient mapmaker based on the maximum likelihood mapmaking equation:
    (P^T N^{-1} P) m = P^T N^{-1} d
or in more simple terms we solve
    A x = b
"""

import os
from concurrent.futures import ThreadPoolExecutor
from time import time as _time

import healpy as hp
import matplotlib.pyplot as plt
import numba as nb
import numpy as np
from scipy.sparse.linalg import LinearOperator, cg

import globals as g
import spectra
import utils
from argparser import args


def ang2pix_threaded(nside, lon, lat, nworkers=6):
    out = np.empty(lon.shape, dtype=np.int32)
    bounds = np.linspace(0, lon.shape[0], nworkers + 1).astype(int)
    def work(k):
        s, e = bounds[k], bounds[k+1]
        out[s:e] = hp.ang2pix(nside, lon[s:e], lat[s:e], lonlat=True)
    with ThreadPoolExecutor(nworkers) as ex:
        list(ex.map(work, range(nworkers)))
    return out



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


@nb.njit(parallel=True, fastmath=True)
def _b_kernel_2d(d, pointing, inv_sigma2, n_pix, ifg_size):
    """fossil: pointing is (ifg_size, n_data)"""
    b = np.zeros((n_pix, ifg_size), dtype=np.float64)
    for x_i in nb.prange(ifg_size):
        for i in range(pointing.shape[1]):
            b[pointing[x_i, i], x_i] += d[i, x_i] * inv_sigma2
    return b


@nb.njit(parallel=True, fastmath=True)
def _b_kernel_3d(d, pointing, inv_sigma2, n_pix, ifg_size, n_ifgs):
    """firas: pointing is (ifg_size, n_data, n_ifgs)"""
    b = np.zeros((n_pix, ifg_size), dtype=np.float64)
    for x_i in nb.prange(ifg_size):
        for ifg_i in range(n_ifgs):
            for i in range(pointing.shape[1]):
                b[pointing[x_i, i, ifg_i], x_i] += d[i, x_i] * inv_sigma2
    return b


def calculate_b_numba(d, pointing, sigma, n_pix, ifg_size, n_ifgs):
    sigma_arr = np.asarray(sigma)
    if sigma_arr.ndim != 0:
        return _calculate_b_numba_sigma_vector(d, pointing, sigma_arr, n_pix, ifg_size)
    inv_sigma2 = 1.0 / float(sigma_arr) ** 2
    if pointing.ndim == 2:
        return _b_kernel_2d(d, pointing, inv_sigma2, n_pix, ifg_size)
    return _b_kernel_3d(d, pointing, inv_sigma2, n_pix, ifg_size, n_ifgs)

# @nb.njit(parallel=True, fastmath=True)
def A_dot_x(x, pointing, sigma, n_pix):
    """
    Implementation of A @ x.
    """
    x = x.reshape((n_pix, g.IFG_SIZE[args.sim_type]))

    Pm = np.zeros((pointing.shape[1], g.IFG_SIZE[args.sim_type]), dtype=np.float64)

    if args.sim_type == "fossil":
        for x_i in range(g.IFG_SIZE[args.sim_type]):
            Pm[:, x_i] = x[pointing[x_i], x_i]
    elif args.sim_type == "firas":
        for x_i in range(g.IFG_SIZE[args.sim_type]):
            for ifg_i in range(g.N_IFGS):
                Pm[:, x_i] += x[pointing[x_i, :, ifg_i], x_i]

    # 2) Weight
    NPm = Pm / sigma**2

    # 3) Scatter back
    Ax = np.zeros_like(x)
    # for ifg_i in range(g.N_IFGS):
    #     for x_i in range(g.IFG_SIZE[args.sim_type]):
    #         for data_point_i in range(pointing.shape[0]):
    #             if args.sim_type == "fossil":
    #                 pix = pointing[data_point_i, x_i]
    #             elif args.sim_type == "firas":
    #                 pix = pointing[data_point_i, x_i, ifg_i]
    #             Ax[pix, x_i] += NPm[data_point_i, x_i]
    if args.sim_type == "fossil":
        for x_i in range(g.IFG_SIZE[args.sim_type]):
            Ax[:, x_i] = np.bincount(pointing[x_i], weights=NPm[:, x_i], minlength=n_pix)
    elif args.sim_type == "firas":
        for x_i in range(g.IFG_SIZE[args.sim_type]):
            for ifg_i in range(g.N_IFGS):
                Ax[:, x_i] += np.bincount(pointing[x_i, :, ifg_i], weights=NPm[:, x_i],
                                          minlength=n_pix)

    # add regularization term to Ax
    Ax += 1e-2 * x

    return Ax.ravel()

# def getP():
    # P = csc_matrix((hits, (rows, cols)), shape=(n_data, n_pix * n_ifg))

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

def cg_scipy(pointing):
    n_ifgs = g.N_IFGS if args.sim_type == "firas" else 1
    n_pix = g.NPIX[args.sim_type]
    A = LinearOperator((n_pix * ifg_size * n_ifgs, n_pix * ifg_size * n_ifgs), matvec=lambda x:
                       A_dot_x(x, pointing, sigma, n_pix))

    x, _ = cg(A, b.ravel(), x0=x0.ravel(), maxiter=1000, tol=1e-5, M=invv_map)
    return x

def test_symmetry(pointing):
    """
    Test that the operator A is symmetric.
    """
    x = np.random.rand(n_pix * ifg_size)
    y = np.random.rand(n_pix * ifg_size)

    Ax = A_dot_x(x, pointing, sigma, n_pix)
    Ay = A_dot_x(y, pointing, sigma, n_pix)

    lhs = np.dot(x.ravel(), Ay.ravel())
    rhs = np.dot(y.ravel(), Ax.ravel())

    print(f"Symmetry test: lhs={lhs}, rhs={rhs}, diff={lhs - rhs}")

@nb.njit(parallel=True, fastmath=True)
def compute_inv_variance_map_2d(pointing, sigma, n_pix, ifg_size, n_ifgs):
    invv = np.zeros((n_pix, ifg_size), dtype=np.float64)

    invsigma2 = 1.0 / sigma**2

    for x_i in nb.range(ifg_size):
        for data_point_i in range(pointing.shape[1]):
            pix = pointing[x_i, data_point_i]

            if sigma.ndim == 0:
                invv[pix, x_i] += invsigma2
            else:
                invv[pix, x_i] += invsigma2[data_point_i, x_i]

    return invv

@nb.njit(parallel=True, fastmath=True)
def compute_inv_variance_map_3d(pointing, sigma, n_pix, ifg_size, n_ifgs):
    invv = np.zeros((n_pix, ifg_size), dtype=np.float64)

    invsigma2 = 1.0 / sigma**2

    for ifg_i in range(n_ifgs):
        for x_i in nb.prange(ifg_size):
            for data_point_i in range(pointing.shape[1]):
                pix = pointing[x_i, data_point_i, ifg_i]

                if sigma.ndim == 0:
                    invv[pix, x_i] += invsigma2
                else:
                    invv[pix, x_i] += invsigma2[data_point_i, x_i, ifg_i]

    return invv


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

    if not os.path.exists(f"../output/data/{args.sim_type}/pix_nside{g.NSIDE[args.sim_type]}.npy"):
        t0 = utils.log_step("ang2pix", t0, args.run_name)
        pix = ang2pix_threaded(g.NSIDE[args.sim_type], ecl_lon, ecl_lat)
        pix = np.swapaxes(pix, 0, 1)
        pix = np.ascontiguousarray(pix)
        np.save(f"../output/data/{args.sim_type}/pix_nside{g.NSIDE[args.sim_type]}.npy",
                pix)
    else:
        pix = np.load(f"../output/data/{args.sim_type}/pix_nside{g.NSIDE[args.sim_type]}.npy", mmap_mode="r")

    # print("Testing symmetry")
    # test_symmetry(pix)

    t0 = utils.log_step("calculate n_ifgs", t0, args.run_name)
    n_ifgs = g.N_IFGS if args.sim_type == "firas" else 1
    t0 = utils.log_step("calculate b_numba", t0, args.run_name)
    b = calculate_b_numba(ifgs, pix, sigma, n_pix, ifg_size, n_ifgs)

    # The CG operator uses pixel-major flattening: idx = pix * n_ifg + ifg.
    # Build x0 in 2D and ravel to avoid IFG-major ordering mistakes.
    x0 = np.zeros((n_pix, ifg_size), dtype=np.float64)
    for i in range(ifg_size):
        x0[:, i] = hp.read_map(f"../output/noise_weighted/{args.sim_type}/ifg_maps/{i:04d}.fits")

    # NaNs mark pixels never hit by the scan, so the mask is per-pixel and
    # survives the rFFT along the ifg axis.
    bad_pix = np.isnan(x0).any(axis=1)
    x0 = np.nan_to_num(x0, nan=0.0)

    t0 = utils.log_step("compute_inv_variance_map", t0, args.run_name)
    if args.sim_type == "fossil":
        invv_map = compute_inv_variance_map_2d(pix, sigma, n_pix, ifg_size, n_ifgs)
    else:
        invv_map = compute_inv_variance_map_3d(pix, sigma, n_pix, ifg_size, n_ifgs)

    t0 = utils.log_step("preconditioned_conjugate_gradient", t0, args.run_name)
    x = preconditioned_conjugate_gradient(b, pix, sigma, invv_map,
                                            x=x0, t0=t0, npix=n_pix)

    x = x.reshape((n_pix, ifg_size))
    m = np.fft.rfft(x, axis=1).real
    m[bad_pix, :] = hp.UNSEEN

    frequencies = spectra.generate_frequencies(simtype=args.sim_type,
                                               nfreq=g.SPEC_SIZE[args.sim_type])

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
