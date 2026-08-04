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
import scipy.fft as fft
from erfa import ErfaWarning

import globals as g
import sims.dust_map as dust_map
import sims.noise as noise
import sims.scanning_strategy as ss
import utils
from argparser import args

# ignore far future warning
warnings.filterwarnings('ignore', category=ErfaWarning)

with open(f"../output/profiling/{args.run_name}.txt", "w") as f:
    f.write("Profiling output for FOSSIL simulation\n")
    f.write(f"Number of workers used: {args.nworkers}\n")
    f.write("=" * 50 + "\n")

t0 = _time()
t00 = _time()

data_dir = "../output/data/fossil"
if not os.path.exists(f"{data_dir}/ecl_lat.npy"):
    ecl_lon, ecl_lat = ss.create_pointings(args)
    t0 = utils.log_step("create_pointings", t0, args.run_name)
else:
    ecl_lon = np.load(f"{data_dir}/ecl_lon.npy", mmap_mode="r")
    t0 = utils.log_step("load ecl_lon", t0, args.run_name)
    ecl_lat = np.load(f"{data_dir}/ecl_lat.npy", mmap_mode="r")
    t0 = utils.log_step("load ecl_lat", t0, args.run_name)
    
dust_map_Mjy, frequencies, sed = dust_map.sim_dust("fossil", t0, args.run_name)
t0 = utils.log_step("sim_dust", t0, args.run_name)
# TODO: problem should be somewhere after here

sed_ifg = fft.irfft(sed)
t0 = utils.log_step("irfft", t0, args.run_name)

ifg = np.multiply.outer(dust_map_Mjy, sed_ifg)
t0 = utils.log_step("multiply dust map", t0, args.run_name)
ifg = np.roll(ifg, 180, axis=1)
t0 = utils.log_step("roll (index remap prep)", t0, args.run_name)
ifg = ifg.real
t0 = utils.log_step("real", t0, args.run_name)

if args.plots == "debug":
    dust = np.multiply.outer(dust_map_Mjy, sed)

    dust_map_dir = "../output/sims/fossil/dust_maps"
    args_list = [(frequencies[nui], dust[:, nui], dust_map_dir) for nui in range(len(frequencies))]
    t0 = utils.log_step("prepare args_list for save_maps", t0, args.run_name)

    for freq, dust_map_i, out_dir in args_list:
        utils.save_maps(freq, dust_map_i, out_dir, write_png=True)
    print(f"Saved dust maps to {dust_map_dir}.")
    t0 = utils.log_step("save_dust_maps", t0, args.run_name)

# now we frankenstein the IFGs together
# col_idx = np.arange(pix_ecl.shape[1])
# get the pixel at the NSIDE that the dust map is at for each IFG
nside_dust = hp.get_nside(dust_map_Mjy)
pix_ecl = hp.ang2pix(nside_dust, ecl_lon, ecl_lat, lonlat=True)
ifg_scanning = ifg[pix_ecl, np.arange(pix_ecl.shape[1])] 
t0 = utils.log_step("ifg_scanning indexing", t0, args.run_name)

n = random.randrange(ifg_scanning.shape[0])
if args.plots == "debug":
    ifg_dir = "../output/sims/fossil/ifgs"

    plt.plot(ifg_scanning[n])
    plt.title(f"IFG {n}")
    plt.ylabel("Interferogram")
    plt.savefig(f"{ifg_dir}/{n}.png")
    plt.close()
    print(f"Saved IFG {n} to {ifg_dir}.")

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
    plt.savefig(g.PIX_HIT_DIR / f"fossil_{n}.png")
    plt.close()

    print(f"Saved pixel hit map for IFG {n} to {g.PIX_HIT_DIR}.")

# add white noise
if args.noise:
    noise, sigma = noise.white_noise(ifg_scanning.shape[0], simtype="fossil", args=args,
                                    signal=ifg_scanning, ifg=False)
    t0 = utils.log_step("white_noise", t0, args.run_name)

if args.noise:
    ifg_final = ifg_scanning + noise
else:
    ifg_final = ifg_scanning

if args.plots == "debug" and args.noise:
    plt.plot(ifg_final[n], alpha=0.5, label="Signal + Noise")
    plt.plot(ifg_scanning[n], alpha=0.5, label="Signal")
    plt.plot(noise[n], alpha=0.5, label="Noise")
    
    plt.title(f"IFG {n} with noise")
    plt.ylabel("Interferogram")
    plt.legend()
    plt.savefig(ifg_dir / f"{n}_with_noise.png")

    plt.ylim(-0.001, 0.001)
    plt.savefig(ifg_dir / f"{n}_with_noise_zoomed.png")

    print(f"Saved IFG {n} with noise to {ifg_dir}.")


np.save(f"{g.DATA_DIR}/ifgs.npy", ifg_final)
if args.noise:
    np.save(f"{g.DATA_DIR}/noise.npy", sigma)
print(f"Saved IFGs, pixel indices, and noise to {g.DATA_DIR}.")

with open(f"../output/profiling/{args.run_name}.txt", "a") as f:
    f.write("=" * 50 + "\n")
    f.write(f"Total time for FOSSIL simulation: {(_time() - t00)/60:.2f} min\n")

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
import scipy.fft as fft
from erfa import ErfaWarning

import globals as g
import sims.dust_map as dust_map
import sims.noise as noise
import sims.scanning_strategy as ss
import utils
from argparser import args

# ignore far future warning
warnings.filterwarnings('ignore', category=ErfaWarning)

with open(f"../output/profiling/{args.run_name}.txt", "w") as f:
    f.write("Profiling output for FOSSIL simulation\n")
    f.write(f"Number of workers used: {args.nworkers}\n")
    f.write("=" * 50 + "\n")

t0 = _time()
t00 = _time()

data_dir = "../output/data/fossil"
if not os.path.exists(f"{data_dir}/ecl_lat.npy"):
    ecl_lon, ecl_lat = ss.create_pointings(args)
    t0 = utils.log_step("create_pointings", t0, args.run_name)
else:
    ecl_lon = np.load(f"{data_dir}/ecl_lon.npy", mmap_mode="r")
    t0 = utils.log_step("load ecl_lon", t0, args.run_name)
    ecl_lat = np.load(f"{data_dir}/ecl_lat.npy", mmap_mode="r")
    t0 = utils.log_step("load ecl_lat", t0, args.run_name)
    
dust_map_Mjy, frequencies, sed = dust_map.sim_dust("fossil", t0, args.run_name)
t0 = utils.log_step("sim_dust", t0, args.run_name)
# TODO: problem should be somewhere after here

sed_ifg = fft.irfft(sed)
t0 = utils.log_step("irfft", t0, args.run_name)

ifg = np.multiply.outer(dust_map_Mjy, sed_ifg)
t0 = utils.log_step("multiply dust map", t0, args.run_name)
ifg = np.roll(ifg, 180, axis=1)
t0 = utils.log_step("roll", t0, args.run_name)
ifg = ifg.real
t0 = utils.log_step("real", t0, args.run_name)

if args.plots == "debug":
    dust = np.multiply.outer(dust_map_Mjy, sed)

    dust_map_dir = "../output/sims/fossil/dust_maps"
    args_list = [(frequencies[nui], dust[:, nui], dust_map_dir) for nui in range(len(frequencies))]
    t0 = utils.log_step("prepare args_list for save_maps", t0, args.run_name)

    for freq, dust_map_i, out_dir in args_list:
        utils.save_maps(freq, dust_map_i, out_dir, write_png=True)
    print(f"Saved dust maps to {dust_map_dir}.")
    t0 = utils.log_step("save_dust_maps", t0, args.run_name)

# now we frankenstein the IFGs together
# col_idx = np.arange(pix_ecl.shape[1])
# get the pixel at the NSIDE that the dust map is at for each IFG
nside_dust = hp.get_nside(dust_map_Mjy)
pix_ecl = hp.ang2pix(nside_dust, ecl_lon, ecl_lat, lonlat=True)
ifg_scanning = np.zeros((pix_ecl.shape[0], ifg.shape[1]))
_ifg_scanning = ifg[pix_ecl, np.arange(pix_ecl.shape[1])]
for col_idx in range(pix_ecl.shape[1]):
    for row_idx in range(pix_ecl.shape[0]):
        ifg_scanning[row_idx, col_idx] = ifg[pix_ecl[row_idx, col_idx], col_idx]  
np.testing.assert_array_almost_equal(ifg_scanning, _ifg_scanning)
t0 = utils.log_step("ifg_scanning indexing", t0, args.run_name)

n = random.randrange(ifg_scanning.shape[0])
if args.plots == "debug":
    ifg_dir = "../output/sims/fossil/ifgs"

    plt.plot(ifg_scanning[n])
    plt.title(f"IFG {n}")
    plt.ylabel("Interferogram")
    plt.savefig(f"{ifg_dir}/{n}.png")
    plt.close()
    print(f"Saved IFG {n} to {ifg_dir}.")

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
    plt.savefig(g.PIX_HIT_DIR / f"fossil_{n}.png")
    plt.close()

    print(f"Saved pixel hit map for IFG {n} to {g.PIX_HIT_DIR}.")

# add white noise
if args.noise:
    noise, sigma = noise.white_noise(ifg_scanning.shape[0], simtype="fossil", args=args,
                                    signal=ifg_scanning, ifg=False)
    t0 = utils.log_step("white_noise", t0, args.run_name)

if args.noise:
    ifg_final = ifg_scanning + noise
else:
    ifg_final = ifg_scanning

if args.plots == "debug" and args.noise:
    plt.plot(ifg_final[n], alpha=0.5, label="Signal + Noise")
    plt.plot(ifg_scanning[n], alpha=0.5, label="Signal")
    plt.plot(noise[n], alpha=0.5, label="Noise")
    
    plt.title(f"IFG {n} with noise")
    plt.ylabel("Interferogram")
    plt.legend()
    plt.savefig(ifg_dir / f"{n}_with_noise.png")

    plt.ylim(-0.001, 0.001)
    plt.savefig(ifg_dir / f"{n}_with_noise_zoomed.png")

    print(f"Saved IFG {n} with noise to {ifg_dir}.")


np.save(f"{g.DATA_DIR}/ifgs.npy", ifg_final)
if args.noise:
    np.save(f"{g.DATA_DIR}/noise.npy", sigma)
print(f"Saved IFGs, pixel indices, and noise to {g.DATA_DIR}.")

with open(f"../output/profiling/{args.run_name}.txt", "a") as f:
    f.write("=" * 50 + "\n")
    f.write(f"Total time for FOSSIL simulation: {(_time() - t00)/60:.2f} min\n")