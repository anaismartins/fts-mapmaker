"""
Maximum likelihood mapmaker that solves the equation
    (P^T M^T N^{-1} M P) m = P^T M^T N^{-1} d,
assuming there is only white noise i.e. N is diagonal, which means the equation reduces to
    m = sum (d / sigma ^2) / sum (1 / sigma^2).
"""

from time import time as _time

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np

import globals as g
import spectra
import utils
from argparser import args

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
    ifgs = np.roll(ifgs, -360, axis=1)

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
pix_grid = hp.ang2pix(g.NSIDE[args.sim_type], ecl_lon, ecl_lat, lonlat=True)  

t0 = utils.log_step("compute w_noise", t0, args.run_name)
w_noise = 1.0 / sigma**2

t0 = utils.log_step("compute numerator and denominator", t0, args.run_name)
if args.sim_type == "fossil":
    for x_i in range(g.IFG_SIZE[args.sim_type]):
        vals = ifgs[:, x_i] * w_noise
        
        pix  = pix_grid[:, x_i]
        # bincount returns length npix; fill the column x_i for numerator/denominator
        numerator[:, x_i] = np.bincount(pix, weights=vals, minlength=g.NPIX[args.sim_type])
        denominator[:, x_i] = np.bincount(pix, weights=np.ones_like(vals) * 1.0/(sigma**2),
                                          minlength=g.NPIX[args.sim_type])
elif args.sim_type == "firas":
    for ifg_i in range(g.N_IFGS):
        for x_i in range(g.IFG_SIZE[args.sim_type]):
            vals = ifgs[:, x_i] * w_noise

            pix = pix_grid[:, x_i, ifg_i]
            
            # bincount returns length npix; fill the column x_i for numerator/denominator
            numerator[:, x_i] += np.bincount(pix, weights=vals, minlength=g.NPIX[args.sim_type])
            denominator[:, x_i] += np.bincount(pix, weights=np.ones_like(vals) * 1.0/(sigma**2),
                                               minlength=g.NPIX[args.sim_type])
else:
    raise ValueError(f"Unknown sim_type: {args.sim_type}")

t0 = utils.log_step("compute m_ifg", t0, args.run_name)
mask = denominator == 0

m_ifg = np.zeros((g.NPIX[args.sim_type], g.IFG_SIZE[args.sim_type]), dtype=float)
m_ifg[~mask] = numerator[~mask] / denominator[~mask]
m_ifg[mask] = np.nan

m = np.abs(np.fft.rfft(m_ifg, axis=1))
phase = np.angle(np.fft.rfft(m_ifg, axis=1))

if args.sim_type == "fossil":
    nfreq = 129
elif args.sim_type == "firas":
    nfreq = 257
frequencies = spectra.generate_frequencies(simtype=args.sim_type, nfreq=nfreq)

path = f"../output/white_noise/{args.sim_type}/"
for nui, freq in enumerate(frequencies):
    if g.FITS:
        hp.write_map(f"{path}{int(freq):04d}.fits", m[:, nui], overwrite=True, dtype=np.float64)

    if g.PNG:
        hp.mollview(m[:, nui], title=f"{int(freq):04d} GHz", unit="MJy/sr", min=0, max=50,
                    xsize=2000, coord=["E", "G"])
        plt.savefig(f"{path}{int(freq):04d}.png")
        plt.close()
print(f"Saved maps to {path}.")
