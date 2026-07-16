"""
This script generates simulated data for a modern FTS experiment.
It assumes the same speeds as FIRAS, but without summing up on-board IFGs which are then
telemetered, i.e. we assume that on-board = telemetered IFG.

NB!!! Should be run on a machine with quite a bit of RAM, as it generates all of the simulations at
once, and uses around 400 - 500 GB at peak.
"""

import argparse
import os
import random
import warnings
from pathlib import Path
from multiprocessing import Pool, cpu_count
from time import time
import utils

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
from erfa import ErfaWarning

import globals as g
import sims.utils as sims
from sims.scanning_strategy import calculate_batch
from sims.fossil_utils import create_pointings

from time import time as _time

with open("../output/profiling.txt", "w") as f:
    f.write("Profiling output for FOSSIL simulation\n")
    f.write("=" * 50 + "\n")

def log_step(label, t_start):
    t = _time()
    with open("../output/profiling.txt", "a") as f:
        f.write(f"[{label}] took {t - t_start:.2f} s\n")
    return t

# ignore far future warning
warnings.filterwarnings('ignore', category=ErfaWarning)

# set up line arguments
parser = argparse.ArgumentParser(description="Simulate scanning strategy for FOSSIL.")
parser.add_argument("--verbose", "-v", action="store_true", help="Increase output verbosity.")
parser.add_argument(
    "--no-plots",
    action="store_true",
    help="Skip plotting and writing PNG diagnostics.",
)
parser.add_argument(
    "--workers",
    type=int,
    default=None,
    help="Override the number of worker processes used for scanning batches.",
)

args = parser.parse_args()

plot_outputs = not args.no_plots

print("Simulating scanning strategy for FOSSIL...")

if plot_outputs:
    for directory in [DUST_MAP_DIR, IFG_DIR, PIX_HIT_DIR, DATA_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

def available_cpu_count():
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return cpu_count()

pointing_cache = g.DATA_DIR / f"sim_pointing.npz"

if not pointing_cache.exists():
    create_pointings()
else:
    print("Loading existing pointings...")
    pointing = np.load(pointing_cache)

    pix_ecl = pointing["pix"]
    ecl_lon = pointing["lon"]
    ecl_lat = pointing["lat"]

    if pix_ecl.ndim == 1:
        pix_ecl = np.array(np.split(pix_ecl, ecl_lon.shape[0]))

    print(f"Loaded pointings from {pointing_cache}")

t0 = _time()
print("Simulating dust map...")
dust_map_Mjy, frequencies, sed = sims.sim_dust("fossil")
t0 = log_step("sim_dust", t0)
# TODO: problem should be somewhere after here

print("Working on SEDs...")
dust = dust_map_Mjy[:, np.newaxis] * sed[np.newaxis, :]
bb = utils.planck(frequencies, temp=2.7)
t0 = log_step("planck + dust multiplication", t0)

print("Generating interferograms...")
ifg = np.fft.irfft(dust - bb, axis=1)
t0 = log_step("irfft", t0)
ifg = np.roll(ifg, 180, axis=1)
ifg = ifg.real
print(f"Generated interferogram cube with shape {ifg.shape}")

if plot_outputs:
    # save maps for each frequency
    for nui in range(len(frequencies)):
        spectral_map = dust_map_Mjy * sed[nui]
        hp.mollview(
            spectral_map,
            title=f"Spectral map at {frequencies[nui]:.2f} GHz",
            unit="MJy/sr",
            xsize=2000,
            coord=["E", "G"],
            min=0,
            max=50,
        )
        plt.savefig(DUST_MAP_DIR / f"{int(frequencies[nui]):04d}.png")
        plt.close()
    print(f"Saved dust maps to {DUST_MAP_DIR} ----------------------------------------------------")

# now we frankenstein the IFGs together
col_idx = np.arange(pix_ecl.shape[1])
ifg_scanning = ifg[pix_ecl, col_idx]
t0 = log_step("ifg_scanning indexing", t0)

print(f"Frankensteined IFGs together with shape {ifg_scanning.shape}.")

n = random.randrange(ifg_scanning.shape[0])
if plot_outputs:
    plt.plot(ifg_scanning[n])
    plt.title(f"IFG {n}")
    plt.ylabel("Interferogram")
    plt.savefig(IFG_DIR / f"{n}.png")
    plt.close()
    print(f"Saved IFG {n} to {IFG_DIR} -----------------------------------------------------------")

# plot pixels hit on a map
# Create a two-panel figure: full sky + zoomed view
if plot_outputs:
    fig = plt.figure(figsize=(16, 6))

    print(f"Pixels hit: {np.unique(pix_ecl[n])}")
    npix = hp.nside2npix(g.NSIDE["fossil"])
    map_pix = np.bincount(pix_ecl[n], minlength=npix)
    ax1 = plt.subplot(1, 2, 1)
    hp.mollview(map_pix, coord="E", title="Pixels hit", cmap="Reds", hold=True)
    hp.projplot(ecl_lon[n], ecl_lat[n], coord="E", color="green", lonlat=True, marker="x")

    ax1.set_position([0.05, 0.1, 0.4, 0.8])
    ax2 = plt.subplot(1, 2, 2)
    hp.gnomview(
        map_pix,
        rot=(ecl_lon[n], ecl_lat[n]),
        title="Pixels hit (gnomonic)",
        cmap="RdYlGn",
        hold=True,
    )
    hp.projplot(
        ecl_lon[n],
        ecl_lat[n],
        coord="E",
        color="blue",
        lonlat=True,
        marker="x",
    )

    current_ax = plt.gca()
    current_ax.ticklabel_format(style="plain", axis="both")
    current_ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.1f}"))
    current_ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.1f}"))
    plt.savefig(PIX_HIT_DIR / f"fossil_{n}.png")
    plt.close()

    print(f"Saved pixel hit map for IFG {n} to {PIX_HIT_DIR} -------------------------------------")

# add white noise
noise, sigma = sims.white_noise(ifg_scanning.shape[0], simtype="fossil", signal=ifg_scanning,
                                ifg=False)
t0 = log_step("white_noise", t0)

ifg_final = ifg_scanning + noise

if plot_outputs:
    plt.plot(ifg_final[n], alpha=0.5, label="Signal + Noise")
    plt.plot(ifg_scanning[n], alpha=0.5, label="Signal")
    plt.plot(noise[n], alpha=0.5, label="Noise")
    
    plt.title(f"IFG {n} with noise")
    plt.ylabel("Interferogram")
    plt.legend()
    plt.savefig(IFG_DIR / f"{n}_with_noise.png")

    plt.ylim(-0.001, 0.001)
    plt.savefig(IFG_DIR / f"{n}_with_noise_zoomed.png")

    print(f"Saved IFG {n} with noise to {IFG_DIR} ------------------------------------------------")


np.savez(f"{DATA_DIR}/ifgs.npz", ifg_final)
np.savez(f"{DATA_DIR}/pointing.npz", pix_ecl)
np.savez(f"{DATA_DIR}/noise.npz", sigma)
print(f"Saved IFGs, pixel indices, and noise to {DATA_DIR} ---------------------------------------")
