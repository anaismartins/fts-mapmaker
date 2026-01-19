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


if not os.path.exists("../output/sim_pointing_fossil.npz"):
    # run for full survey  using parallelization
    n_batches = int(survey_len * 365.25 * obs_eff) # one day batches
    n_workers = min(cpu_count() // 10, n_batches)
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

    # save all pointings
    np.savez("../output/sim_pointing_fossil.npz", pix=pix_ecl, lon=ecl_lon, lat=ecl_lat)
    print("Saved pointings to ../output/sim_pointing_fossil.npz")
else:
    print("Loading existing pointings...")
    pix_ecl = np.load("../output/sim_pointing_fossil.npz")["pix"]
    ecl_lon = np.load("../output/sim_pointing_fossil.npz")["lon"]
    ecl_lat = np.load("../output/sim_pointing_fossil.npz")["lat"]
    print("Loaded pointings from ../output/sim_pointing_fossil.npz")

if args.verbose:
    print(f"Total number of pointings for the whole survey: {len(pix_ecl):,}")
    print("Creating hit map...")
    hit_map = np.bincount(pix_ecl, minlength=hp.nside2npix(g.NSIDE["fossil"]))
    hit_map = hit_map / g.NPIXPERIFG["fossil"]

    print("Generating and saving plot...")
    hp.mollview(hit_map, title="Hit Map for Fossil Scanning", unit="Hits",coord=["E", "G"])
    plt.savefig("../output/hit_maps/scanning_strategy_fossil.png", bbox_inches="tight")
    plt.close()
    print("Saved hit map to ../output/hit_maps/scanning_strategy_fossil.png")

    # save fits file
    hp.write_map(
        "../output/hit_maps/scanning_strategy_fossil.fits",
        hit_map,
        overwrite=True,
    )

print("Simulating dust map...")
dust_map_downgraded_mjy, frequencies, sed = sims.sim_dust()
sed = np.nan_to_num(sed)

print("Generating interferograms...")
spec = dust_map_downgraded_mjy[:, np.newaxis] * sed[np.newaxis, :]
print(f"Shape of spec: {spec.shape}")

ifg = np.fft.irfft(spec, axis=1)
ifg = np.roll(ifg, 360, axis=1)
ifg = ifg.real

print(f"shape of pix_ecl: {pix_ecl.shape}")
# divide pix_ecl into the different ifgs
rest = len(pix_ecl) % g.NPIXPERIFG["fossil"]
print(f"rest: {rest}")
pix_ecl = np.array(np.split(pix_ecl[:-rest], g.NPIXPERIFG["fossil"])).T
print(f"shape of pix_ecl after reshaping: {pix_ecl.shape}")

# now we frankenstein the IFGs together
ifg_scanning = np.zeros((len(pix_ecl), g.IFG_SIZE))
for i in range(g.NPIXPERIFG["fossil"]):
    for pix_i, pix in enumerate(pix_ecl[:, i]):
        ifg_scanning[pix_i, i] = ifg[pix, i]

print(f"Shape of ifg_scanning: {ifg_scanning.shape}")

n = random.randint(0, ifg_scanning.shape[0])

print(f"Plotting IFG {n}...")

plt.plot(ifg_scanning[n])
plt.title(f"IFG {n}")
plt.ylabel("Interferogram")
plt.savefig(f"../output/sims/ifgs_modern/{n}.png")
plt.close()

# plot pixels hit on a map
print(f"Pixels hit: {np.unique(pix_ecl[n])}")
npix = hp.nside2npix(g.NSIDE["fossil"])
map_pix = np.bincount(pix_ecl[n], minlength=npix)
hp.mollview(map_pix, coord="E", title="Pixels hit", cmap="Reds")
hp.projplot(
    ecl_lon[n],
    ecl_lat[n],
    coord="E",
    color="green",  # "blue",
    lonlat=True,
    marker="x",
)
plt.savefig(f"../output/pix_hits/fossil_{n}.png")

# add white noise
noise, sigma = sims.white_noise(ifg_scanning.shape[0])

ifg_scanning = ifg_scanning + noise
print(f"Shape of noise: {noise.shape} and shape of sigma: {sigma.shape}")

np.savez("../output/ifgs_modern.npz", ifg=ifg_scanning, pix=pix_ecl, sigma=sigma)
print("Saved IFGs")
