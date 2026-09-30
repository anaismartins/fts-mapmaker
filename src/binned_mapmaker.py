"""
Maximum likelihood mapmaker that solves the equation
    (P^T M^T N^{-1} M P) m = P^T M^T N^{-1} d,
assuming there is only white noise i.e. N is diagonal, which means the equation reduces to
    m = sum (d / sigma ^2) / sum (1 / sigma^2).
"""

import os
from time import time as _time

import healpy as hp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numba as nb
import numpy as np

import globals as g
import spectra
import utils
from argparser import args


@nb.njit(parallel=True)
def accumulate_fossil(ifgs, pix_grid, weight, n_pix):
    n_channels = ifgs.shape[1]
    numerator = np.zeros((n_pix, n_channels), dtype=np.float64)
    denominator = np.zeros((n_pix, n_channels), dtype=np.float64)

    for x_i in nb.prange(n_channels):
        for row in range(ifgs.shape[0]):
            pix = pix_grid[row, x_i]
            numerator[pix, x_i] += ifgs[row, x_i] * weight
            denominator[pix, x_i] += 1.0

    return numerator, denominator


with open(f"../output/profiling/{args.run_name}.txt", "w") as f:
    f.write("Profiling output for white noise mapmaker for FOSSIL\n")
    f.write("=" * 50 + "\n")
    f.write(f"{'starting':<40} | ")

t00 = _time()
t0 = _time()

if args.sim_type == "fossil":
    add_on = ""
    folder_add_on = ""
elif args.sim_type == "firas":
    if args.firas_ss:
        add_on = "_firas"
    else:
        add_on = "_fossil"
    folder_add_on = f"/ss{add_on}"
else:
    raise ValueError(f"Unknown sim_type: {args.sim_type}")

t0 = utils.log_step("load ifgs", t0, args.run_name)
ifgs = np.load(f"../output/data/{args.sim_type}/ifgs{add_on}.npy", mmap_mode="r")
t0 = utils.log_step("load pix", t0, args.run_name)
ecl_lon = np.load(f"../output/data/{args.sim_type}/ecl_lon{add_on}.npy", mmap_mode="r")
ecl_lat = np.load(f"../output/data/{args.sim_type}/ecl_lat{add_on}.npy", mmap_mode="r")
if args.noise:
    t0 = utils.log_step("load sigma", t0, args.run_name)
    sigma = np.load(f"../output/data/{args.sim_type}/noise{add_on}.npy", mmap_mode="r")

    t0 = utils.log_step("compute w_noise", t0, args.run_name)
    w_noise = 1.0 / sigma**2
else:
    w_noise = 1

if args.sim_type == "firas":
    t0 = utils.log_step("divide ifgs by N_IFGS", t0, args.run_name)
    ifgs = ifgs / g.N_IFGS
    
t0 = utils.log_step("initialize numerator and denominator", t0, args.run_name)
# how many unique pixels are there?
numerator = np.zeros((g.NPIX[args.sim_type], g.IFG_SIZE[args.sim_type]), dtype=float)
denominator = np.zeros_like(numerator, dtype=float)
# Vectorized accumulation: loop over IFG sample index (usually much smaller
# than the number of IFGs) and use np.bincount to accumulate values per pixel.
# This avoids the expensive Python-level loop over all IFGs and is much faster.
t0 = utils.log_step("ang2pix", t0, args.run_name)

if not os.path.exists(f"../output/data/{args.sim_type}/pix_nside{g.NSIDE[args.sim_type]}{add_on}.npy"):
    pix_grid = hp.ang2pix(g.NSIDE[args.sim_type], ecl_lon, ecl_lat, lonlat=True)  
    np.save(f"../output/data/{args.sim_type}/pix_nside{g.NSIDE[args.sim_type]}{add_on}.npy",
            pix_grid)
else:
    pix_grid = np.load(f"../output/data/{args.sim_type}/pix_nside{g.NSIDE[args.sim_type]}{add_on}.npy",
                        mmap_mode="r")

t0 = utils.log_step("compute numerator and denominator", t0, args.run_name)
if args.sim_type == "fossil":
    numerator, denominator = accumulate_fossil(ifgs, pix_grid, w_noise, g.NPIX[args.sim_type])
    denominator *= w_noise

elif args.sim_type == "firas":
    for x_i in range(g.IFG_SIZE[args.sim_type]):
        vals = ifgs[:, x_i] * w_noise
        for ifg_i in range(g.N_IFGS):
            pix = pix_grid[:, x_i, ifg_i]
            
            # bincount returns length npix; fill the column x_i for numerator/denominator
            numerator[:, x_i] += np.bincount(pix, weights=vals, minlength=g.NPIX[args.sim_type])
            hits = np.bincount(pix, minlength=g.NPIX[args.sim_type])
            denominator[:, x_i] += hits * w_noise
else:
    raise ValueError(f"Unknown sim_type: {args.sim_type}")

t0 = utils.log_step("create_mask", t0, args.run_name)
mask = denominator == 0

t0 = utils.log_step("initialize m_ifg", t0, args.run_name)
m_ifg = np.zeros((g.NPIX[args.sim_type], g.IFG_SIZE[args.sim_type]), dtype=float)
t0 = utils.log_step("compute m_ifg", t0, args.run_name)
m_ifg[~mask] = numerator[~mask] / denominator[~mask]
t0 = utils.log_step("set empty to nan", t0, args.run_name)
m_ifg[mask] = np.nan

for nui in range(g.IFG_SIZE[args.sim_type]):
    if g.FITS:
        hp.write_map(f"../output/binned/{args.sim_type}{folder_add_on}/ifg_maps/{nui:04d}.fits",
                     m_ifg[:, nui], overwrite=True, dtype=np.float64)
    if g.PNG:
        hp.mollview(m_ifg[:, nui], title=f"IFG {nui:04d}", unit="MJy/sr", min=0, max=50, xsize=2000,
                    coord=["E", "G"])
        plt.savefig(f"../output/binned/{args.sim_type}{folder_add_on}/ifg_maps/{nui:04d}.png")
        plt.close()

# Keep only the real spectral component to match the mapmaking convention.
m = np.fft.rfft(m_ifg, axis=1).real

if args.sim_type == "fossil":
    nfreq = 129
elif args.sim_type == "firas":
    nfreq = 257
frequencies = spectra.generate_frequencies(simtype=args.sim_type, nfreq=nfreq)

path = f"../output/binned/{args.sim_type}{folder_add_on}/maps/"
for nui, freq in enumerate(frequencies):
    if g.FITS:
        hp.write_map(f"{path}{int(freq):04d}.fits", m[:, nui], overwrite=True, dtype=np.float64)

    if g.PNG:
        hp.mollview(m[:, nui], title=f"{int(freq):04d} GHz", unit="MJy/sr", min=0, max=50,
                    xsize=2000, coord=["E", "G"])
        plt.savefig(f"{path}{int(freq):04d}.png")
        plt.close()
print(f"Saved maps to {path}.")

with open(f"../output/profiling/{args.run_name}.txt", "a") as f:
    f.write(f"{(_time() - t0):0.2f}\n")
    f.write("=" * 50 + "\n")
    f.write(f"Total time for white noise mapmaker: {(_time() - t00)/60:.2f} min\n")