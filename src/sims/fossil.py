"""
This script generates simulated data for a modern FTS experiment.
It assumes the same speeds as FIRAS, but without summing up on-board IFGs which are then
telemetered, i.e. we assume that on-board = telemetered IFG.

NB!!! Should be run on a machine with quite a bit of RAM, as it generates all of the simulations at
once, and uses around 400 - 500 GB at peak.
"""

import os
import random
import warnings
from time import time as _time

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
from erfa import ErfaWarning

import globals as g
import sims.dust_map as dust_map
import sims.noise as noise
import sims.scanning_strategy as ss
import spectra
import utils
from argparser import args

# ignore far future warning
warnings.filterwarnings('ignore', category=ErfaWarning)

with open(f"../output/profiling/{args.run_name}.txt", "w") as f:
    f.write("Profiling output for FOSSIL simulation\n")
    f.write(f"Number of workers used: {args.workers}\n")
    f.write("=" * 50 + "\n")

t0 = _time()
t00 = _time()
if not os.path.exists(f"{g.DATA_DIR}/pointing.npy"):
    pix_ecl, ecl_lon, ecl_lat = ss.create_pointings(args)
    t0 = utils.log_step("create_pointings", t0, args.run_name)
else:
    pix_ecl = np.load(g.DATA_DIR / "pointing.npy")
    ecl_lon = np.load(g.DATA_DIR / "ecl_lon.npy")
    ecl_lat = np.load(g.DATA_DIR / "ecl_lat.npy")
    t0 = utils.log_step("load_pointings", t0, args.run_name)

    if pix_ecl.ndim == 1:
        pix_ecl = np.array(np.split(pix_ecl, ecl_lon.shape[0]))
    
dust_map_Mjy, frequencies, sed = dust_map.sim_dust("fossil", t0, args.run_name)
t0 = utils.log_step("sim_dust", t0, args.run_name)
# TODO: problem should be somewhere after here

dust = dust_map_Mjy[:, np.newaxis] * sed[np.newaxis, :]
bb = spectra.planck(frequencies, temp=2.7)
t0 = utils.log_step("planck + dust multiplication", t0, args.run_name)

ifg = np.fft.irfft(dust)# - bb, axis=1)
t0 = utils.log_step("irfft", t0, args.run_name)
ifg = np.roll(ifg, 180, axis=1)
ifg = ifg.real
t0 = utils.log_step("roll", t0, args.run_name)

if args.plots == "debug":
    # save maps for each frequency
    for nui in range(len(frequencies)):
        spectral_map = dust_map_Mjy * sed[nui]
        hp.mollview(spectral_map, title=f"Spectral map at {frequencies[nui]:.2f} GHz",
                    unit="MJy/sr", xsize=2000, coord=["E", "G"], min=0, max=50)
        plt.savefig(g.DUST_MAP_DIR / f"{int(frequencies[nui]):04d}.png")
        plt.close()
    print(f"Saved dust maps to {g.DUST_MAP_DIR}.")
    t0 = utils.log_step("save_dust_maps", t0, args.run_name)

# now we frankenstein the IFGs together
col_idx = np.arange(pix_ecl.shape[1])
ifg_scanning = ifg[pix_ecl, col_idx]
t0 = utils.log_step("ifg_scanning indexing", t0, args.run_name)

n = random.randrange(ifg_scanning.shape[0])
if args.plots == "debug":
    plt.plot(ifg_scanning[n])
    plt.title(f"IFG {n}")
    plt.ylabel("Interferogram")
    plt.savefig(g.IFG_DIR / f"{n}.png")
    plt.close()
    print(f"Saved IFG {n} to {g.IFG_DIR}.")

# plot pixels hit on a map
# Create a two-panel figure: full sky + zoomed view
if args.plots == "debug" or args.plots == "paper_only":
    fig = plt.figure(figsize=(16, 6))

    row_pix = pix_ecl[n]
    row_lon = ecl_lon[n]
    row_lat = ecl_lat[n]
    lon_center = float(np.mean(row_lon))
    lat_center = float(np.mean(row_lat))

    print(f"Pixels hit: {np.unique(row_pix).size} unique pixels by IFG {n}.")
    npix = hp.nside2npix(g.NSIDE["fossil"])
    map_pix = np.bincount(row_pix, minlength=npix)
    vmax = max(1, int(map_pix.max()))
    ax1 = plt.subplot(1, 2, 1)
    hp.mollview(map_pix, coord="E", title="Pixels hit", cmap="Reds", min=0, max=vmax, hold=True)
    hp.projplot(row_lon, row_lat, coord="E", color="green", lonlat=True, marker=".", ms=1)

    ax1.set_position([0.05, 0.1, 0.4, 0.8])
    ax2 = plt.subplot(1, 2, 2)
    hp.gnomview(
        map_pix,
        rot=(lon_center, lat_center, 0),
        title="Pixels hit (gnomonic)",
        cmap="RdYlGn",
        min=0,
        max=vmax,
        coord="E",
        hold=True,
    )
    hp.projplot(
        row_lon,
        row_lat,
        coord="E",
        color="blue",
        lonlat=True,
        marker=".",
        ms=1,
    )

    current_ax = plt.gca()
    current_ax.ticklabel_format(style="plain", axis="both")
    current_ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.1f}"))
    current_ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.1f}"))
    plt.savefig(g.PIX_HIT_DIR / f"fossil_{n}.png")
    plt.close()

    print(f"Saved pixel hit map for IFG {n} to {g.PIX_HIT_DIR}.")

# add white noise
noise, sigma = noise.white_noise(ifg_scanning.shape[0], simtype="fossil", signal=ifg_scanning,
                                ifg=False)
t0 = utils.log_step("white_noise", t0, args.run_name)

ifg_final = ifg_scanning + noise

if args.plots == "debug":
    plt.plot(ifg_final[n], alpha=0.5, label="Signal + Noise")
    plt.plot(ifg_scanning[n], alpha=0.5, label="Signal")
    plt.plot(noise[n], alpha=0.5, label="Noise")
    
    plt.title(f"IFG {n} with noise")
    plt.ylabel("Interferogram")
    plt.legend()
    plt.savefig(g.IFG_DIR / f"{n}_with_noise.png")

    plt.ylim(-0.001, 0.001)
    plt.savefig(g.IFG_DIR / f"{n}_with_noise_zoomed.png")

    print(f"Saved IFG {n} with noise to {g.IFG_DIR}.")


np.save(f"{g.DATA_DIR}/ifgs.npy", ifg_final)
np.save(f"{g.DATA_DIR}/noise.npy", sigma)
print(f"Saved IFGs, pixel indices, and noise to {g.DATA_DIR}.")

with open(f"../output/profiling/{args.run_name}.txt", "a") as f:
    f.write("=" * 50 + "\n")
    f.write(f"Total time for FOSSIL simulation: {(_time() - t00)/60:.2f} min\n")
