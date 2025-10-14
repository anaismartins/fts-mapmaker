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

total_time = 55.36  # seconds
n_ifgs = 16
flyback_time = 0.42  # seconds
time_per_ifg = total_time / n_ifgs  # seconds
time_per_ifg_on_source = time_per_ifg - flyback_time  # seconds

speed_deg_per_min = 3.5
speed = speed_deg_per_min / 60  # degrees per second

# Initialize array to hold ecl_lat for all IFGs
ecl_lats = np.zeros((n_ifgs, len(ecl_lat), npixperifg), dtype=float)

# Compute starting positions for each IFG
t1 = time.time()
for ifg in range(n_ifgs):
    print(f"Computing latitudes for IFG {ifg+1}/{n_ifgs}")
    # Initial position for this IFG
    start_offset = (speed * total_time / 2) * scan_dir
    flyback_offset = speed * flyback_time * scan_dir * ifg
    ecl_lat_init = (
        ecl_lat
        - start_offset
        + flyback_offset
        + speed * time_per_ifg_on_source * scan_dir * ifg
    )

    # Fill in pixel positions for this IFG
    for pix in range(npixperifg):
        ecl_lats[ifg, :, pix] = (
            ecl_lat_init + speed * time_per_ifg_on_source * scan_dir * pix / npixperifg
        )
        # adjust latitudes to be in the range [-90, 90]
        ecl_lats[ifg, :, pix][ecl_lats[ifg, :, pix] < -90] = (
            -ecl_lats[ifg, :, pix][ecl_lats[ifg, :, pix] < -90] - 180
        )
        ecl_lats[ifg, :, pix][ecl_lats[ifg, :, pix] > 90] = (
            180 - ecl_lats[ifg, :, pix][ecl_lats[ifg, :, pix] > 90]
        )
t2 = time.time()
print(f"Time taken for computing the latitudes: {t2-t1} seconds")

t1 = time.time()
pix_ecl = np.zeros((n_ifgs, len(ecl_lat), npixperifg), dtype=int)
for ifg in range(n_ifgs):
    print(f"Computing pixel indices for IFG {ifg+1}/{n_ifgs}")
    for pix_i in range(npixperifg):
        pix_ecl[ifg, :, pix_i] = hp.ang2pix(
            g.NSIDE, ecl_lon, ecl_lats[ifg, :, pix_i], lonlat=True
        )
t2 = time.time()
print(f"Time taken for computing the pixel indices: {t2-t1} seconds")

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
    ifgs = np.zeros((n_ifgs, len(pix_ecl[0]), npixperifg))
    for ifg_idx in range(n_ifgs):
        for pix_idx in range(npixperifg):
            # Assign IFG values for each pixel index
            ifgs[ifg_idx, :, pix_idx] = ifg[pix_ecl[ifg_idx, :, pix_idx]]

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
    plt.show()
