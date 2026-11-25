"""
This script generates simulated data for a modern FTS experiment.
It assumes the same speeds as FIRAS, but without summing up on-board IFGs which are then
telemetered, i.e. we assume that on-board = telemetered IFG.

NB!!! Should be run on a machine with quite a bit of RAM, as it generates all of the simulations at
once, and uses around 400 - 500 GB at peak.
"""

import random
import warnings
from multiprocessing import Pool
from time import time as _time

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import scipy.fft as fft
from erfa import ErfaWarning

import globals as g
import sims.dust_map as dust_map
import sims.noise as noise
import sims.scanning_strategy as ss
import paths
import utils
import config
from config import cfg

# ignore far future warning
warnings.filterwarnings('ignore', category=ErfaWarning)

if cfg.sim_type != "fossil":
    raise ValueError(f"{cfg.sim_name} is a {cfg.sim_type} simulation, not fossil.")
# record the parameters of this version next to its data (read back by the mapmakers)
config.save_record(paths.data_dir(), config.RECORD_NAME)

with open(cfg.log, "w") as f:
    f.write("Profiling output for FOSSIL simulation\n")
    f.write(f"Number of workers used: {cfg.nworkers}\n")
    f.write("=" * 50 + "\n")
    f.write(f"{'starting':<40} | ")

t0 = _time()
t00 = _time()

if not paths.pointing_file("ecl_lat").exists():
    t0 = utils.log_step("create_pointings", t0, cfg.log)
    ecl_lon, ecl_lat = ss.create_pointings()
else:
    t0 = utils.log_step("load ecl_lon", t0, cfg.log)
    ecl_lon = np.load(paths.pointing_file("ecl_lon"), mmap_mode="r")
    t0 = utils.log_step("load ecl_lat", t0, cfg.log)
    ecl_lat = np.load(paths.pointing_file("ecl_lat"), mmap_mode="r")

t0 = utils.log_step("sim_dust", t0, cfg.log)
dust_map_Mjy, frequencies, sed = dust_map.sim_dust("fossil", t0, cfg.log)

t0 = utils.log_step("irfft", t0, cfg.log)
sed_ifg = fft.irfft(sed)

if cfg.dust_nside is not None:
    t0 = utils.log_step("ud_grade dust map", t0, cfg.log)
    dust_map_Mjy = hp.ud_grade(dust_map_Mjy, cfg.dust_nside)

# truth maps at the output resolution, used by compare.py
dust_map_dir = paths.sim_dir("dust_maps")
t0 = utils.log_step("prepare args_list for save_maps", t0, cfg.log)
truth = hp.ud_grade(dust_map_Mjy, cfg.nside)
args_list = [(frequencies[nui], truth * sed[nui], dust_map_dir) for nui in range(len(frequencies))]

t0 = utils.log_step("save_dust_maps", t0, cfg.log)
with Pool(processes=cfg.nworkers) as pool:
    list(pool.starmap(utils.save_maps, args_list))
print(f"Saved dust maps to {dust_map_dir}.")

# now we frankenstein the IFGs together
t0 = utils.log_step("prepare shift", t0, cfg.log)
n_cols = sed_ifg.shape[0]

# col_idx = np.arange(pix_ecl.shape[1])
# get the pixel at the NSIDE that the dust map is at for each IFG
t0 = utils.log_step("get nside", t0, cfg.log)
nside_dust = hp.get_nside(dust_map_Mjy)
t0 = utils.log_step("ang2pix dust", t0, cfg.log)
pix_ecl = utils.ang2pix_cached(nside_dust, ecl_lon, ecl_lat,
                               paths.pointing_file(f"pix_nside{nside_dust}"), nworkers=cfg.nworkers)
t0 = utils.log_step("ang2pix fossil", t0, cfg.log)
if nside_dust == cfg.nside:
    # dust map and fossil output share the same NSIDE, so the pixel indices are identical
    pix_ecl_fossil = pix_ecl
else:
    pix_ecl_fossil = utils.ang2pix_cached(cfg.nside, ecl_lon, ecl_lat,
                                          paths.pointing_file(f"pix_nside{cfg.nside}"),
                                          nworkers=cfg.nworkers)

# plot hit map
if cfg.plots == "debug" or cfg.plots == "paper_only":
    npix = hp.nside2npix(cfg.nside)
    map_pix = np.bincount(pix_ecl_fossil.flatten(), minlength=npix)
    hp.mollview(map_pix, coord=["E", "G"], title="FOSSIL Scanning Strategy Hit map",
                unit="Number of hits", min=0)
    hit_map_dir = paths.hit_map_dir()
    plt.savefig(hit_map_dir / "scanning_strategy.png")
    plt.close()
    hp.write_map(hit_map_dir / "scanning_strategy.fits", map_pix, overwrite=True)
    print(f"Saved pixel hit map for all IFGs to {hit_map_dir}.")

t0 = utils.log_step("ifg_scanning indexing", t0, cfg.log)
ifg_scanning = (dust_map_Mjy[pix_ecl] * sed_ifg).real

n = random.randrange(ifg_scanning.shape[0])
ifg_dir = paths.sim_dir("ifgs")
if cfg.plots == "debug":
    plt.plot(ifg_scanning[n])
    plt.title(f"IFG {n}")
    plt.ylabel("Interferogram")
    plt.savefig(f"{ifg_dir}/{n}.png")
    plt.close()
    print(f"Saved IFG {n} to {ifg_dir}.")

if cfg.plots == "debug" or cfg.plots == "paper_only":
    # plot showing how a full IFG is built from frankenstein-ing together different IFGs. shows where each pixel starts and ends in the final IFG.
    pix_ifg = pix_ecl_fossil[n]
    for i in range(len(pix_ifg)):
        if pix_ifg[i] != pix_ifg[i - 1]:
            plt.axvline(i, color="red", alpha=0.5)

    plt.plot(ifg_scanning[n])
    # plt.vlines(, ymin=ifg_scanning[n].min(), ymax=ifg_scanning[n].max(), color="red", alpha=0.5, label="Pixel boundaries")
    # plt.title(f"IFG {n} with pixel boundaries")
    plt.ylabel("Interferogram")
    plt.xlabel("IFG sample index")
    # plt.legend()
    plt.savefig(f"{ifg_dir}/{n}_with_pixel_boundaries.png")
    plt.close()
    print(f"Saved IFG {n} with pixel boundaries to {ifg_dir}.")

# plot pixels hit on a map
# Create a two-panel figure: full sky + zoomed view
if cfg.plots == "debug" or cfg.plots == "paper_only":
    fig = plt.figure(figsize=(16, 6))

    row_pix = pix_ecl_fossil[n]
    row_lon = ecl_lon[n]
    row_lat = ecl_lat[n]
    lon_center = float(np.mean(row_lon))
    lat_center = float(np.mean(row_lat))

    print(f"Pixels hit: {np.unique(row_pix).size} unique pixels by IFG {n}.")
    npix = hp.nside2npix(cfg.nside)
    map_pix = np.bincount(row_pix, minlength=npix)
    vmax = max(1, int(map_pix.max()))
    ax1 = plt.subplot(1, 2, 1)
    hp.mollview(map_pix, coord="E", title="Pixels hit", cmap="RdYlGn", min=0, max=vmax, hold=True)
    hp.projplot(lon_center, lat_center, coord="E", color="blue", lonlat=True, marker="x", ms=10)

    ax1.set_position([0.05, 0.1, 0.4, 0.8])
    ax2 = plt.subplot(1, 2, 2)
    hp.gnomview(map_pix, rot=(lon_center, lat_center, 0), title="Pixels hit", cmap="RdYlGn", min=0,
                max=vmax, coord="E", hold=True)
    hp.projplot(lon_center, lat_center, coord="E", color="blue", lonlat=True, marker="x", ms=10)

    current_ax = plt.gca()
    current_ax.ticklabel_format(style="plain", axis="both")
    current_ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.1f}"))
    current_ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.1f}"))

    pix_hit_dir = paths.sim_dir("pix_hits")
    plt.savefig(f"{pix_hit_dir}/fossil_{n}.png")
    plt.close()

    print(f"Saved pixel hit map for IFG {n} to {pix_hit_dir}.")

# add white noise
if cfg.noise:
    t0 = utils.log_step("white_noise", t0, cfg.log)
    noise, sigma = noise.white_noise(ifg_scanning.shape[0], simtype="fossil")

if cfg.noise:
    ifg_final = ifg_scanning + noise
else:
    ifg_final = ifg_scanning

if cfg.plots == "debug" and cfg.noise:

    plt.plot(ifg_final[n], alpha=0.5, label="Signal + Noise")
    plt.plot(ifg_scanning[n], alpha=0.5, label="Signal")
    plt.plot(noise[n], alpha=0.5, label="Noise")
    
    plt.title(f"IFG {n} with noise")
    plt.ylabel("Interferogram")
    plt.legend()
    plt.savefig(f"{ifg_dir}/{n}_with_noise.png")

    plt.ylim(-0.001, 0.001)
    plt.savefig(f"{ifg_dir}/{n}_with_noise_zoomed.png")

    print(f"Saved IFG {n} with noise to {ifg_dir}.")



np.save(paths.data_file("ifgs"), ifg_final)
if cfg.noise:
    t0 = utils.log_step("save noise", t0, cfg.log)
    np.save(paths.data_file("noise"), sigma)
print(f"Saved IFGs, pixel indices, and noise to {paths.data_dir()}. Number of generated IFGs was " 
      f"{ifg_final.shape[0]}.")

with open(cfg.log, "a") as f:
    f.write("=" * 50 + "\n")
    f.write(f"Total time for FOSSIL simulation: {(_time() - t00)/60:.2f} min\n")
