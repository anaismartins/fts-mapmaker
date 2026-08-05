"""
This script generates simulated data for a FIRAS-like experiment.
This means it assumes the FIRAS scanning speed, as well as the on-board coadding of IFGs, and it assumes data is taken in the short slow mode.
"""

import os
from time import time as _time

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np

import globals as g
import sims.dust_map as dust_map
import sims.noise as noise
import utils
from argparser import args

with open(f"../output/profiling/{args.run_name}.txt", "w") as f:
    f.write("Profiling output for FIRAS simulation\n")
    f.write("=" * 50 + "\n")
    label = "sim dust"
    f.write(f"{label:<35} | ")

t0 = _time()
t00 = _time()

dust_map_downgraded_mjy, frequencies, sed = dust_map.sim_dust("firas", t0, args.run_name)
sed = np.nan_to_num(sed)

t0 = utils.log_step("irfft", t0, args.run_name)
sed_ifg = np.fft.irfft(sed, norm="ortho")

t0 = utils.log_step("multiply dust map", t0, args.run_name)
ifg = np.multiply.outer(dust_map_downgraded_mjy, sed_ifg)

ifg = np.roll(ifg, 360, axis=1)
ifg = ifg.real

t0 = utils.log_step("load_sky_data", t0, args.run_name)
user = os.environ["USER"]
data_path = f"/mn/stornext/d5/data/{user}/firas-reanalysis/FIRAS-Pass5/data/preprocessed_sky_ll.npz"
sky_data = np.load(data_path, allow_pickle=True)

mtm_speed = sky_data["mtm_speed"][:]
mtm_length = sky_data["mtm_length"][:]
ss_filter = (mtm_speed == 0) & (mtm_length == 0)
ecl_lat = sky_data["ecl_lat"][ss_filter]
ecl_lon = sky_data["ecl_lon"][ss_filter]

total_time = 55.36  # seconds
flyback_time = 0.42  # seconds
time_per_ifg = total_time / g.N_IFGS  # seconds
time_per_ifg_on_source = time_per_ifg - flyback_time  # seconds

speed_deg_per_min = 3.5
speed = speed_deg_per_min / 60  # degrees per second

t0 = utils.log_step("initialize_arrays", t0, args.run_name)
ecl_lats = np.zeros((len(ecl_lat), g.NPIXPERIFG["firas"], g.N_IFGS), dtype=float)

# Create arrays for IFG and pixel indices
ifg_indices = np.arange(g.N_IFGS)  # shape: (N_IFGS,)
pix_indices = np.arange(g.NPIXPERIFG["firas"])  # shape: (NPIXPERIFG,)

# Broadcast to create meshgrid for vectorized computation
# ifg_grid shape: (1, 1, N_IFGS), pix_grid shape: (1, NPIXPERIFG, 1)
ifg_grid = ifg_indices[np.newaxis, np.newaxis, :]
pix_grid = pix_indices[np.newaxis, :, np.newaxis]

# Vectorized computation for all positions at once
start_offset = speed * total_time / 2
flyback_offset = speed * flyback_time * ifg_grid
time_offset = speed * time_per_ifg_on_source * ifg_grid
pix_offset = speed * time_per_ifg_on_source * pix_grid / g.NPIXPERIFG["firas"]

t0 = utils.log_step("compute_latitudes", t0, args.run_name)
# Broadcast ecl_lat to match the shape (N_ecl_lat, 1, 1)
ecl_lat_broadcast = ecl_lat[:, np.newaxis, np.newaxis]

# Compute all latitudes at once
ecl_lats = ecl_lat_broadcast - start_offset + flyback_offset + time_offset + pix_offset

# make ecl_lons have the same shape as ecl_lats. ecl_lon now has shape of the number of recorded IFGs
# we want it to have shape (that, npixperifg, n_ifgs) as ecl_lats with copies of the longitudes along the second and third dimensions
ecl_lons = np.array(np.broadcast_to(ecl_lon[:, np.newaxis, np.newaxis], ecl_lats.shape))

# Adjust latitudes to be in the range [-90, 90] (vectorized)
mask_low = ecl_lats < -90
ecl_lats[mask_low] = -ecl_lats[mask_low] - 180
ecl_lons[mask_low] = 180 - ecl_lons[mask_low]
mask_high = ecl_lats > 90
ecl_lats[mask_high] = 180 - ecl_lats[mask_high]
ecl_lons[mask_high] = 180 - ecl_lons[mask_high]

t0 = utils.log_step("compute_pixel_indices", t0, args.run_name)
nside_dust = hp.get_nside(dust_map_downgraded_mjy)
pix_ecl = np.zeros((len(ecl_lat), g.NPIXPERIFG["firas"], g.N_IFGS), dtype=int)
pix_ecl_firas = np.zeros((len(ecl_lat), g.NPIXPERIFG["firas"], g.N_IFGS), dtype=int)
# Vectorized computation of pixel indices for all IFGs and pixels
for ifg_i in range(g.N_IFGS):
    pix_ecl[:, :, ifg_i] = hp.ang2pix(nside_dust,
                                      np.broadcast_to(ecl_lon[:, None],
                                                      ecl_lats[:, :, ifg_i].shape),
                                      ecl_lats[:, :, ifg_i], lonlat=True)
    pix_ecl_firas[:, :, ifg_i] = hp.ang2pix(g.NSIDE["firas"],
                                            np.broadcast_to(ecl_lon[:, None],
                                                            ecl_lats[:, :, ifg_i].shape),
                                            ecl_lats[:, :, ifg_i], lonlat=True)

npix = hp.nside2npix(nside_dust)
if args.plots == "debug":
    t0 = utils.log_step("save_hit_map", t0, args.run_name)
    hit_map = np.bincount(pix_ecl_firas.flatten(), minlength=npix) / g.N_IFGS / g.NPIXPERIFG["firas"]
    mask = hit_map == 0
    hit_map[mask] = np.nan
    if g.PNG:
        hp.mollview(hit_map, title="FIRAS Scanning Strategy Hit Map",
                    unit="Hits per on-board IFG per pixel", min=0, max=332, xsize=2000,
                    coord=["E", "G"])

        plt.savefig("../output/hit_maps/scanning_strategy_firas_sim.png", facecolor=None,
                    bbox_inches="tight")
        plt.close()
    if g.FITS:
        hp.write_map("../output/hit_maps/scanning_strategy_firas_sim.fits", hit_map, overwrite=True,
                    dtype=np.float64)
    print("Saved hit map of the scanning strategy to ../output/hit_maps/.")
    

t0 = utils.log_step("frankenstein_ifgs", t0, args.run_name)
# Combine each of the 16 IFGs, filling all 512 points for each IFG
ifgs = np.zeros((pix_ecl.shape[0], g.NPIXPERIFG["firas"], g.N_IFGS))  # 16 x npix x 512
# Vectorized assignment to speed up frankensteining IFGs

for ifg_i in range(g.N_IFGS):
    ifgs[:, :, ifg_i] = ifg[pix_ecl[:, :, ifg_i], np.arange(g.NPIXPERIFG["firas"])]

# and lastly we add the 16 ifgs together
total_ifg = np.sum(ifgs, axis=2)

if args.plots == "debug":
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    n = np.random.randint(0, total_ifg.shape[0])
    ax[0].plot(ifgs[n], alpha=0.5)
    ax[0].set_title(f"IFGs for pixel {n}")
    ax[0].set_ylabel("Interferogram")
    ax[1].plot(total_ifg[n])
    ax[1].set_title(f"Total IFG for pixel {n}")
    ax[1].set_ylabel("Interferogram")
    plt.tight_layout()
    plt.savefig(f"../output/sims/firas/ifgs/{n}.png")
    plt.close()
    print(f"Saved IFG {n} to ../output/sims/firas/ifgs/{n}.png.")

# plot pixels hit on a map
pix_to_map = np.zeros(g.N_IFGS * g.NPIXPERIFG["firas"], dtype=int)
for ifg_i in range(g.N_IFGS):
    pix_to_map[ifg_i * g.NPIXPERIFG["firas"] : (ifg_i + 1) * g.NPIXPERIFG["firas"]] = pix_ecl[n, :, ifg_i]
map_pix = np.bincount(pix_to_map, minlength=npix)

if args.plots == "debug" or args.plots == "paper_only":
    # Create a two-panel figure: full sky + zoomed view
    fig = plt.figure(figsize=(16, 6))

    # Left panel: full-sky mollview
    ax1 = plt.subplot(1, 2, 1)
    hp.mollview(map_pix, title="Pixels hit for one interferogram (Full Sky)", unit="Hits", min=0,
                max=g.N_IFGS * g.NPIXPERIFG["firas"], xsize=2000, coord="E", cmap="Reds", hold=True,
                sub=(1, 2, 1), format="%d")
    hp.projplot(ecl_lon[n], ecl_lat[n], coord="E", color="green", lonlat=True, marker="x",
                markersize=5)
    # Adjust left panel position to center it better
    ax1.set_position([0.05, 0.1, 0.4, 0.8])

    # Right panel: zoomed gnomonic view centered on the pixel
    ax2 = plt.subplot(1, 2, 2)
    hp.gnomview(map_pix, rot=(ecl_lon[n], ecl_lat[n]), title="Zoomed view (2° radius)", unit="Hits",
        min=0, # max=map_pix.max(),
        max=g.FITS * g.NPIXPERIFG["firas"],  # Maximum possible hits
        xsize=800, coord="E", cmap="Reds", reso=1.0,  # resolution in arcmin
        hold=True, sub=(1, 2, 2), format="%d",  # Format colorbar as integers
        )
    hp.projplot(ecl_lon[n], ecl_lat[n], coord="E", color="green", lonlat=True, marker="x",
                markersize=10)

    # Format the axes tick labels to avoid scientific notation on the right panel
    current_ax = plt.gca()
    current_ax.ticklabel_format(style="plain", axis="both")
    current_ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.1f}"))
    current_ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.1f}"))

    plt.savefig(f"../output/sims/firas/pix_hits/{n}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved pixel hit map for IFG {n} to ../output/sims/firas/pix_hits/{n}.png.")

# add white noise
if args.noise:
    noise, sigma = noise.white_noise(total_ifg.shape[0], simtype="firas")
    total_ifg = total_ifg + noise

if args.plots == "debug" and args.noise:
    plt.plot(total_ifg[n], alpha=0.5, label="Signal + Noise")
    plt.plot(np.sum(ifgs[n], axis=1), alpha=0.5, label="Signal")
    plt.plot(noise[n], alpha=0.5, label="Noise")
    plt.title(f"IFG {n} with noise")
    plt.ylabel("Interferogram")
    plt.legend()
    plt.savefig(f"../output/sims/firas/ifgs/{n}_with_noise.png")
    plt.close()
    print(f"Saved IFG {n} with noise to ../output/sims/firas/ifgs/{n}_with_noise.png.")

np.save("../output/data/firas/ifgs.npy", total_ifg)
np.save("../output/data/firas/ecl_lat.npy", ecl_lats)
np.save("../output/data/firas/ecl_lon.npy", ecl_lons)
if args.noise:
    np.save("../output/data/firas/noise.npy", noise)
print("Saved FIRAS IFGs to ../output/data/firas/.")

with open(f"../output/profiling/{args.run_name}.txt", "a") as f:
    f.write("=" * 50 + "\n")
    f.write(f"Total time for FIRAS simulation: {(_time() - t00)/60:.2f} min\n")