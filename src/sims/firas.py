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

dust_map_downgraded_mjy, frequencies, sed = sims.sim_dust("firas")
sed = np.nan_to_num(sed)

spec = dust_map_downgraded_mjy[:, np.newaxis] * sed[np.newaxis, :]
print(f"Shape of spec: {spec.shape}")

ifg = np.fft.irfft(spec, axis=1)
ifg = np.roll(ifg, 360, axis=1)
ifg = ifg.real

print(f"Shape of ifg: {ifg.shape}")

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

# Initialize array to hold ecl_lat for all IFGs
ecl_lats = np.zeros((len(ecl_lat), g.NPIXPERIFG["firas"], g.N_IFGS), dtype=float)

# Compute starting positions for each IFG
t1 = time.time()
print(f"Computing latitudes for all IFGs (vectorized)")

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

# Broadcast ecl_lat to match the shape (N_ecl_lat, 1, 1)
ecl_lat_broadcast = ecl_lat[:, np.newaxis, np.newaxis]

# Compute all latitudes at once
ecl_lats = (
    ecl_lat_broadcast
    - start_offset
    + flyback_offset
    + time_offset
    + pix_offset
)

# make ecl_lons have the same shape as ecl_lats. ecl_lon now has shape of the number of recorded IFGs
# we want it to have shape (that, npixperifg, n_ifgs) as ecl_lats with copies of the longitudes along the second and third dimensions
ecl_lons = np.array(np.broadcast_to(ecl_lon[:, np.newaxis, np.newaxis], ecl_lats.shape))
# check what it looks like
print(ecl_lons)

# Adjust latitudes to be in the range [-90, 90] (vectorized)
print(f"shapes of ecl_lats and ecl_lon before adjustment: {ecl_lats.shape}, {ecl_lon.shape}")
mask_low = ecl_lats < -90
ecl_lats[mask_low] = -ecl_lats[mask_low] - 180
ecl_lons[mask_low] = 180 - ecl_lons[mask_low]
mask_high = ecl_lats > 90
ecl_lats[mask_high] = 180 - ecl_lats[mask_high]
ecl_lons[mask_high] = 180 - ecl_lons[mask_high]

t2 = time.time()
print(f"Time taken for computing the latitudes: {t2-t1} seconds")

t1 = time.time()
pix_ecl = np.zeros((len(ecl_lat), g.NPIXPERIFG["firas"], g.N_IFGS), dtype=int)
# Vectorized computation of pixel indices for all IFGs and pixels
for ifg_i in range(g.N_IFGS):
    print(f"Computing pixel indices for IFG {ifg_i+1}/{g.N_IFGS}")
    # ecl_lon shape: (N,), ecl_lats[ifg_i] shape: (N, npixperifg)
    # Broadcast ecl_lon to (N, npixperifg) for vectorized ang2pix
    pix_ecl[:, :, ifg_i] = hp.ang2pix(
        g.NSIDE["firas"],
        np.broadcast_to(ecl_lon[:, None], ecl_lats[:, :, ifg_i].shape),
        ecl_lats[:, :, ifg_i],
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
npix = hp.nside2npix(g.NSIDE["firas"])
hit_map = np.bincount(pix_ecl.flatten(), minlength=npix) / g.N_IFGS / g.NPIXPERIFG["firas"]
mask = hit_map == 0
hit_map[mask] = np.nan
if g.PNG:
    hp.mollview(hit_map, title="FIRAS Scanning Strategy Hit Map",
                unit="Hits per on-board IFG per pixel", min=0, max=332, xsize=2000,
                coord=["E", "G"])

    plt.savefig("../output/hit_maps/scanning_strategy_firas.png", facecolor=None,
                bbox_inches="tight")
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
ifgs = np.zeros((pix_ecl.shape[0], g.NPIXPERIFG["firas"], g.N_IFGS))  # 16 x npix x 512
# Vectorized assignment to speed up frankensteining IFGs

# check shapes
print(f"Shape of pix_ecl: {pix_ecl.shape}")
print(f"Shape of ifg: {ifg.shape}")
for ifg_i in range(g.N_IFGS):
    print(f"Frankensteining IFG {ifg_i+1}/{g.N_IFGS}")
    # pix_ecl[ifg_i]: shape (num_pixels, npixperifg)
    # Use advanced indexing to assign all IFG values at once
    ifgs[:, :, ifg_i] = ifg[pix_ecl[:, :, ifg_i], np.arange(g.NPIXPERIFG["firas"])]
t2 = time.time()
print(f"Time taken for frankensteining the IFGs: {t2-t1} seconds")

# and lastly we add the 16 ifgs together
total_ifg = np.sum(ifgs, axis=2)

fig, ax = plt.subplots(2, 1, figsize=(10, 8))
n = np.random.randint(0, total_ifg.shape[0])
ax[0].plot(ifgs[n], alpha=0.5)
ax[0].set_title(f"IFGs for pixel {n}")
ax[0].set_ylabel("Interferogram")
ax[1].plot(total_ifg[n])
ax[1].set_title(f"Total IFG for pixel {n}")
ax[1].set_ylabel("Interferogram")
plt.tight_layout()
plt.savefig(f"../output/sim_firas/ifg/{n}.png")
plt.close()

# plot pixels hit on a map
pix_to_map = np.zeros(g.N_IFGS * g.NPIXPERIFG["firas"], dtype=int)
for ifg_i in range(g.N_IFGS):
    pix_to_map[ifg_i * g.NPIXPERIFG["firas"] : (ifg_i + 1) * g.NPIXPERIFG["firas"]] = pix_ecl[n, :, ifg_i]
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
    max=g.N_IFGS * g.NPIXPERIFG["firas"],  # Maximum possible hits
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
    max=g.FITS * g.NPIXPERIFG["firas"],  # Maximum possible hits
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
