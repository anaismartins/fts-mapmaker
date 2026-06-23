"""
This script generates simulated data for a modern FTS experiment.
It assumes the same speeds as FIRAS, but without summing up on-board IFGs which are then
telemetered, i.e. we assume that on-board = telemetered IFG.
"""

import argparse
import os
import random
import warnings
from multiprocessing import Pool, cpu_count
from time import time

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
from erfa import ErfaWarning

import globals as g
import sims.utils as sims
from sims.scanning_strategy import calculate_batch

# ignore far future warning
warnings.filterwarnings('ignore', category=ErfaWarning)

# set up line arguments
parser = argparse.ArgumentParser(description="Simulate scanning strategy for FOSSIL.")
parser.add_argument("--verbose", "-v", action="store_true", help="Increase output verbosity.")
args = parser.parse_args()

print("Simulating scanning strategy for FOSSIL...")

# instrument parameters
survey_len = 4 # years
survey_time = survey_len * 365.25 * 24 * 3600 # seconds
obs_eff = 0.7


if not os.path.exists("../output/data/sim_pointing_fossil.npz"):
    # run for full survey  using parallelization
    n_batches = int(survey_len * 365.25 * obs_eff) # one day batches
    n_workers = min(cpu_count() // 8, n_batches)
    print(f"\n{'='*60}")
    print(f"Starting parallel processing of {n_batches} batches")
    print(f"Using {n_workers} workers (CPU cores available: {cpu_count()})")
    print(f"{'='*60}\n")

    t_start = time()
    with Pool(n_workers) as pool:
        results = pool.map(calculate_batch, range(n_batches))
    t_end = time()

    print(f"\n{'='*60}")
    print(f"Parallel processing complete!")
    print(f"Total time: {t_end - t_start:.2f} seconds")
    print(f"Average time per batch: {(t_end - t_start)/n_batches:.2f} seconds")
    print(f"{'='*60}\n")

    # Combine results
    print("Combining results from all batches...")
    # extract pix from results
    pix_list, lon_list, lat_list = zip(*results)
    pix_ecl = np.concatenate(pix_list)
    ecl_lon = np.concatenate(lon_list)
    ecl_lat = np.concatenate(lat_list)

    print("Shapes gotten from simulation:")
    print(f"pix {pix_ecl.shape}, lon {ecl_lon.shape}, lat {ecl_lat.shape}")

    # save all pointings
    np.savez("../output/data/sim_pointing_fossil.npz", pix=pix_ecl, lon=ecl_lon, lat=ecl_lat)
    print("Saved pointings to ../output/data/sim_pointing_fossil.npz")
else:
    print("Loading existing pointings...")
    pointing = np.load("../output/data/sim_pointing_fossil.npz")
    pix_ecl = pointing["pix"]
    ecl_lon = pointing["lon"]
    ecl_lat = pointing["lat"]
    print("Loaded pointings from ../output/data/sim_pointing_fossil.npz")

print("Simulating dust map...")
dust_map_Mjy, frequencies, sed = sims.sim_dust("fossil")
sed = np.nan_to_num(sed)

print("Generating interferograms...")
spec = dust_map_Mjy[:, np.newaxis] * sed[np.newaxis, :]
print(f"Generated spectral cube with shape {spec.shape}")

# save maps for each frequency
for nui in range(len(frequencies)):
    hp.mollview(spec[:, nui], title=f"Spectral map at {frequencies[nui]:.2f} GHz", unit="MJy/sr",
                xsize=2000, coord=["E", "G"], min=0, max=50)
    plt.savefig(f"../output/dust_maps/fossil/{int(frequencies[nui]):04d}.png")
    plt.close()

# TODO: problem should be somewhere after here

ifg = np.fft.irfft(spec, axis=1)
ifg = np.roll(ifg, 180, axis=1)
ifg = ifg.real

print(f"Generated interferogram cube with shape {ifg.shape}. pix_ecl has shape: {pix_ecl.shape}")

# now we frankenstein the IFGs together
col_idx = np.arange(pix_ecl.shape[1])
ifg_scanning = ifg[pix_ecl, col_idx]

print(f"Frankensteined IFGs together with shape {ifg_scanning.shape}.")

n = random.randrange(ifg_scanning.shape[0])
print(f"Plotting IFG {n}...")

plt.plot(ifg_scanning[n])
plt.title(f"IFG {n}")
plt.ylabel("Interferogram")
plt.savefig(f"../output/sims/ifgs_fossil/{n}.png")
plt.close()

# plot pixels hit on a map
# Create a two-panel figure: full sky + zoomed view
fig = plt.figure(figsize=(16, 6))

print(f"Pixels hit: {np.unique(pix_ecl[n])}")
npix = hp.nside2npix(g.NSIDE["fossil"])
map_pix = np.bincount(pix_ecl[n], minlength=npix)
ax1 = plt.subplot(1, 2, 1)
hp.mollview(map_pix, coord="E", title="Pixels hit", cmap="Reds", hold=True)
hp.projplot(ecl_lon[n], ecl_lat[n], coord="E", color="green", lonlat=True, marker="x")

ax1.set_position([0.05, 0.1, 0.4, 0.8])
ax2 = plt.subplot(1, 2, 2)
hp.gnomview(map_pix, rot=(ecl_lon[n], ecl_lat[n]), title="Pixels hit (gnomonic)", cmap="Reds",
            hold=True)
hp.projplot(
    ecl_lon[n],
    ecl_lat[n],
    coord="E",
    color="green",
    lonlat=True,
    marker="x",
)

current_ax = plt.gca()
current_ax.ticklabel_format(style="plain", axis="both")
current_ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.1f}"))
current_ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.1f}"))
plt.savefig(f"../output/pix_hits/fossil_{n}.png")
plt.close()

# add white noise
noise, sigma = sims.white_noise(ifg_scanning.shape[0], simtype="fossil", signal=ifg_scanning)

ifg_scanning = ifg_scanning + noise
print(f"Shape of noise: {noise.shape} and shape of sigma: {sigma.shape}")

np.savez(f"../output/ifgs_{g.SIM_TYPE}.npz", ifg=ifg_scanning, pix=pix_ecl, sigma=sigma)
print("Saved IFGs")
