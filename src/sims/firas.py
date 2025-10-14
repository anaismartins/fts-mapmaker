"""
This script generates simulated data for a FIRAS-like experiment.
This means it assumes the FIRAS scanning speed, as well as the on-board coadding of IFGs, and it assumes data is taken in the short slow mode.
"""

import os
import time

import h5py
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np

import globals as g
import sims.utils as sims

dust_map_downgraded_mjy, frequencies, sed = sims.sim_dust()
sed = np.nan_to_num(sed)

spec = dust_map_downgraded_mjy[:, np.newaxis] * sed[np.newaxis, :]

ifg = np.fft.irfft(spec, axis=1)
ifg = np.roll(ifg, 360, axis=1)
ifg = ifg.real

print(f"Shape of ifg: {ifg.shape}")

user = os.environ["USER"]
data_path = f"/mn/stornext/u3/{user}/d5/firas-reanalysis/Commander/commander3/todscripts/firas/data/sky_v4.4.h5"
sky_data = h5py.File(
    data_path,
    "r",
)

mtm_speed = sky_data["df_data"]["mtm_speed"][:]
mtm_length = sky_data["df_data"]["mtm_length"][:]
ss_filter = (mtm_speed == 0) & (mtm_length == 0)
ecl_lat = sky_data["df_data"]["ecl_lat"][ss_filter]
ecl_lon = sky_data["df_data"]["ecl_lon"][ss_filter]
scan_dir = sky_data["df_data"]["scan"][ss_filter]

npixperifg = 512
n_ifgs = 16

total_time = 55.36  # seconds
flyback_time = 0.42  # seconds
time_per_ifg = total_time / n_ifgs  # seconds
time_per_ifg_on_source = time_per_ifg - flyback_time  # seconds

speed_deg_per_min = 3.5
speed = speed_deg_per_min / 60  # degrees per second

# Initialize array to hold ecl_lat for all IFGs
ecl_lats = np.zeros((n_ifgs, len(ecl_lat), npixperifg), dtype=float)

# Compute starting positions for each IFG
t1 = time.time()
for ifg_i in range(n_ifgs):
    print(f"Computing latitudes for IFG {ifg_i+1}/{n_ifgs}")
    # Initial position for this IFG
    start_offset = (speed * total_time / 2) * scan_dir
    flyback_offset = speed * flyback_time * scan_dir * ifg_i
    ecl_lat_init = (
        ecl_lat
        - start_offset
        + flyback_offset
        + speed * time_per_ifg_on_source * scan_dir * ifg_i
    )

    # Fill in pixel positions for this IFG
    for pix in range(npixperifg):
        ecl_lats[ifg_i, :, pix] = (
            ecl_lat_init + speed * time_per_ifg_on_source * scan_dir * pix / npixperifg
        )
        # adjust latitudes to be in the range [-90, 90]
        ecl_lats[ifg_i, :, pix][ecl_lats[ifg_i, :, pix] < -90] = (
            -ecl_lats[ifg_i, :, pix][ecl_lats[ifg_i, :, pix] < -90] - 180
        )
        ecl_lats[ifg_i, :, pix][ecl_lats[ifg_i, :, pix] > 90] = (
            180 - ecl_lats[ifg_i, :, pix][ecl_lats[ifg_i, :, pix] > 90]
        )
t2 = time.time()
print(f"Time taken for computing the latitudes: {t2-t1} seconds")

t1 = time.time()
pix_ecl = np.zeros((n_ifgs, len(ecl_lat), npixperifg), dtype=int)
# Vectorized computation of pixel indices for all IFGs and pixels
for ifg_i in range(n_ifgs):
    print(f"Computing pixel indices for IFG {ifg_i+1}/{n_ifgs}")
    # ecl_lon shape: (N,), ecl_lats[ifg_i] shape: (N, npixperifg)
    # Broadcast ecl_lon to (N, npixperifg) for vectorized ang2pix
    pix_ecl[ifg_i] = hp.ang2pix(
        g.NSIDE,
        np.broadcast_to(ecl_lon[:, None], ecl_lats[ifg_i].shape),
        ecl_lats[ifg_i],
        lonlat=True,
    )
t2 = time.time()
print(f"Time taken for computing the pixel indices: {t2-t1} seconds")

# # save pix_ecl
# np.save("../output/sim_firas/pix_ecl.npy", pix_ecl)
# print("Saved pix_ecl to ../output/sim_firas/pix_ecl.npy")

# print(f"Loading pix_ecl from ../output/sim_firas/pix_ecl.npy")
# pix_ecl = np.load("../output/sim_firas/pix_ecl.npy")

print("Saving hit map")
npix = hp.nside2npix(g.NSIDE)
hit_map = np.bincount(pix_ecl.flatten(), minlength=npix) / n_ifgs / npixperifg
mask = hit_map == 0
hit_map[mask] = np.nan
if g.PNG:
    hp.mollview(
        hit_map,
        title="FIRAS Scanning Strategy Hit Map",
        unit="Hits per on-board IFG per pixel",
        min=0,
        max=np.nanmax(hit_map),
        xsize=2000,
        coord=["E", "G"],
    )

    plt.savefig("../output/hit_maps/scanning_strategy_firas.png")
    plt.close()
if g.FITS:
    hp.write_map(
        "../output/hit_maps/scanning_strategy_firas.fits",
        hit_map,
        overwrite=True,
        # dtype=np.float64,
    )

# Combine each of the 16 IFGs, filling all 512 points for each IFG
t1 = time.time()
ifgs = np.zeros((n_ifgs, pix_ecl.shape[1], npixperifg))  # 16 x npix x 512
# Vectorized assignment to speed up frankensteining IFGs
for ifg_i in range(n_ifgs):
    print(f"Frankensteining IFG {ifg_i+1}/{n_ifgs}")
    # pix_ecl[ifg_i]: shape (num_pixels, npixperifg)
    # Use advanced indexing to assign all IFG values at once
    ifgs[ifg_i, :, :] = ifg[pix_ecl[ifg_i], np.arange(npixperifg)]
t2 = time.time()
print(f"Time taken for frankensteining the IFGs: {t2-t1} seconds")

# and lastly we add the 16 ifgs together
total_ifg = np.sum(ifgs, axis=0)

fig, ax = plt.subplots(2, 1, figsize=(10, 8))
n = np.random.randint(0, total_ifg.shape[0])
ax[0].plot(ifgs[:, n, :].T, alpha=0.5)
ax[0].set_title(f"IFGs for pixel {n}")
ax[0].set_ylabel("Interferogram")
ax[1].plot(total_ifg[n])
ax[1].set_title(f"Total IFG for pixel {n}")
ax[1].set_ylabel("Interferogram")
plt.tight_layout()
plt.savefig(f"../output/sim_firas/ifg/{n}.png")
plt.close()

# plot pixels hit on a map
pix_to_map = np.zeros(n_ifgs * npixperifg, dtype=int)
for ifg_i in range(n_ifgs):
    pix_to_map[ifg_i * npixperifg : (ifg_i + 1) * npixperifg] = pix_ecl[ifg_i, n]
map_pix = np.bincount(pix_to_map, minlength=npix)

# Create a two-panel figure: full sky + zoomed view
fig = plt.figure(figsize=(16, 6))

# Left panel: full-sky mollview
ax1 = plt.subplot(1, 2, 1)
hp.mollview(
    map_pix,
    title=f"Pixels hit for one interferogram (Full Sky)",
    unit="Hits",
    min=0,
    # max=map_pix.max(),
    max=n_ifgs * npixperifg,  # Maximum possible hits
    xsize=2000,
    coord="E",
    cmap="Reds",
    hold=True,
    sub=(1, 2, 1),
    format="%d",  # Format colorbar as integers
)
hp.projplot(
    ecl_lon[n],
    ecl_lat[n],
    coord="E",
    color="green",
    lonlat=True,
    marker="x",
    markersize=5,  # Smaller marker for full-sky view
)
# Adjust left panel position to center it better
ax1.set_position([0.05, 0.1, 0.4, 0.8])

# Right panel: zoomed gnomonic view centered on the pixel
ax2 = plt.subplot(1, 2, 2)
hp.gnomview(
    map_pix,
    rot=(ecl_lon[n], ecl_lat[n]),
    title=f"Zoomed view (2° radius)",
    unit="Hits",
    min=0,
    # max=map_pix.max(),
    max=n_ifgs * npixperifg,  # Maximum possible hits
    xsize=800,
    coord="E",
    cmap="Reds",
    reso=1.0,  # resolution in arcmin
    hold=True,
    sub=(1, 2, 2),
    format="%d",  # Format colorbar as integers
)
hp.projplot(
    ecl_lon[n],
    ecl_lat[n],
    coord="E",
    color="green",
    lonlat=True,
    marker="x",
    markersize=10,  # Reasonable size for zoomed view
)

# Format the axes tick labels to avoid scientific notation on the right panel
current_ax = plt.gca()
current_ax.ticklabel_format(style="plain", axis="both")
current_ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.1f}"))
current_ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.1f}"))

plt.savefig(f"../output/sim_firas/pix_hits/{n}.png", dpi=150, bbox_inches="tight")
plt.close()

# add white noise
noise, sigma = sims.white_noise(total_ifg.shape[0])
total_ifg = total_ifg + noise

np.savez(f"../output/ifgs_firas.npz", ifg=total_ifg, pix=pix_ecl, sigma=sigma)
