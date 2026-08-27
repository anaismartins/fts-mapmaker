"""
This script generates simulated data for a FIRAS-like experiment.
This means it assumes the FIRAS scanning speed, as well as the on-board coadding of IFGs, and it
assumes data is taken in the short slow mode.
!!! Needs to be run on an owl with more than 256GB if using the FOSSIL ss.
"""

import os
from multiprocessing import Pool
from pathlib import Path
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
    f.write(f"{label:<40} | ")

t0 = _time()
t00 = _time()

dust_map_Mjy, frequencies, sed = dust_map.sim_dust("firas", t0, args.run_name)
sed = np.nan_to_num(sed)

if args.firas_ss:
    add_on = "firas"
else:
    add_on = "fossil"

if args.plots == "debug" and not os.path.exists("../output/sims/firas/dust_maps/0544.fits"):
        dust = np.multiply.outer(dust_map_Mjy, sed)

        dust_map_dir = "../output/sims/firas/dust_maps"
        Path(f"{dust_map_dir}").mkdir(parents=True, exist_ok=True)
        t0 = utils.log_step("prepare args_list for save_maps", t0, args.run_name)
        args_list = [(frequencies[nui], dust[:, nui], dust_map_dir, add_on) for nui in
                     range(len(frequencies))]

        t0 = utils.log_step("save_dust_maps", t0, args.run_name)
        with Pool(processes=args.nworkers) as pool:
            list(pool.imap_unordered(utils._save_one_map, args_list))
        print(f"Saved dust maps to {dust_map_dir}.")

t0 = utils.log_step("irfft", t0, args.run_name)
sed_ifg = np.fft.irfft(sed)

t0 = utils.log_step("multiply dust map", t0, args.run_name)
ifg = np.multiply.outer(dust_map_Mjy, sed_ifg)

t0 = utils.log_step("take real part of ifg", t0, args.run_name)
ifg = ifg.real

if args.firas_ss:
    print("Using FIRAS scanning strategy for FIRAS simulation.")
    t0 = utils.log_step("load_sky_data", t0, args.run_name)
    user = os.environ["USER"]

    ecl_lat = np.array([])
    ecl_lon = np.array([])
    for channel in g.FIRAS_CHANNELS:
        data_path = f"/mn/stornext/d5/data/aimartin/firas-reanalysis/FIRAS-Pass5/data/preprocessed_sky_{channel}.npz"
        sky_data = np.load(data_path, allow_pickle=True)

        # mtm_speed = sky_data["mtm_speed"][:]
        # mtm_length = sky_data["mtm_length"][:]
        # ss_filter = (mtm_speed == 0) & (mtm_length == 0)
        ecl_lat = np.append(ecl_lat, sky_data["ecl_lat"])#[ss_filter]
        ecl_lon = np.append(ecl_lon, sky_data["ecl_lon"])#[ss_filter]
else:
    print("Using FOSSIL scanning strategy for FIRAS simulation.")
    # mmap + slice a single column instead of loading the full (n_rows, 256) arrays into RAM
    ecl_lat = np.load("../output/data/fossil/ecl_lat.npy", mmap_mode="r")
    ecl_lon = np.load("../output/data/fossil/ecl_lon.npy", mmap_mode="r")

    ecl_lat = np.array(ecl_lat[:, g.NPIXPERIFG["fossil"]//2])
    ecl_lon = np.array(ecl_lon[:, g.NPIXPERIFG["fossil"]//2])

total_time = 55.36  # seconds
flyback_time = 0.42  # seconds
time_per_ifg = total_time / g.N_IFGS  # seconds
time_per_ifg_on_source = time_per_ifg - flyback_time  # seconds

speed_deg_per_min = 3.5
speed = speed_deg_per_min / 60  # degrees per second

n_rows = len(ecl_lat)

# Create arrays for IFG and pixel indices
ifg_indices = np.arange(g.N_IFGS)  # shape: (N_IFGS,)
pix_indices = np.arange(g.NPIXPERIFG["firas"])  # shape: (NPIXPERIFG,)

# Broadcast to create meshgrid for vectorized computation
# ifg_grid shape: (1, 1, N_IFGS), pix_grid shape: (1, NPIXPERIFG, 1)
ifg_grid = ifg_indices[np.newaxis, np.newaxis, :]
pix_grid = pix_indices[np.newaxis, :, np.newaxis]

# Vectorized computation for all positions at once (rows-independent, so computed once)
start_offset = speed * total_time / 2
flyback_offset = speed * flyback_time * ifg_grid
time_offset = speed * time_per_ifg_on_source * ifg_grid
pix_offset = speed * time_per_ifg_on_source * pix_grid / g.NPIXPERIFG["firas"]
lat_offset = -start_offset + flyback_offset + time_offset + pix_offset  # (1, NPIXPERIFG, N_IFGS)

t0 = utils.log_step("get dust NSIDE", t0, args.run_name)
nside_dust = hp.get_nside(dust_map_Mjy)
npix_firas = hp.nside2npix(g.NSIDE["firas"])

# Everything below scales with n_rows, which can be tens of millions for the FOSSIL scanning
# strategy. Rather than allocating (n_rows, NPIXPERIFG, N_IFGS) arrays in memory (which can
# require several TB and cause swapping), process rows in chunks and stream the results to disk
# via memory-mapped .npy files.
chunk_size = max(1, min(args.chunk_size, n_rows))
n_chunks = int(np.ceil(n_rows / chunk_size))
print(f"Processing {n_rows} rows in {n_chunks} chunk(s) of up to {chunk_size} rows.")

data_dir = Path("../output/data/firas")
data_dir.mkdir(parents=True, exist_ok=True)
total_ifg_path = data_dir / f"ifgs_{add_on}.npy"
ecl_lat_out_path = data_dir / f"ecl_lat_{add_on}.npy"
ecl_lon_out_path = data_dir / f"ecl_lon_{add_on}.npy"

total_ifg_mm = np.lib.format.open_memmap(total_ifg_path, mode="w+", dtype=np.float64,
                                          shape=(n_rows, g.NPIXPERIFG["firas"]))
ecl_lat_mm = np.lib.format.open_memmap(ecl_lat_out_path, mode="w+", dtype=np.float64,
                                        shape=(n_rows, g.NPIXPERIFG["firas"], g.N_IFGS))
ecl_lon_mm = np.lib.format.open_memmap(ecl_lon_out_path, mode="w+", dtype=np.float64,
                                        shape=(n_rows, g.NPIXPERIFG["firas"], g.N_IFGS))

hit_map_counts = np.zeros(npix_firas, dtype=np.int64)

if args.noise:
    # sigma only depends on the total number of rows (not the chunk size), so compute it once
    sigma = noise.compute_sigma(n_rows, simtype="firas")

# pick one row up front (for debug plots) and remember which chunk it falls in
n = np.random.randint(0, n_rows) if args.plots in ("debug", "paper_only") else None

for chunk_i in range(n_chunks):
    row_start = chunk_i * chunk_size
    row_end = min(row_start + chunk_size, n_rows)

    t0 = utils.log_step(f"chunk {chunk_i + 1}/{n_chunks}: compute_ecl_lats", t0, args.run_name)
    ecl_lat_chunk = ecl_lat[row_start:row_end, np.newaxis, np.newaxis]
    ecl_lats_chunk = ecl_lat_chunk + lat_offset  # (chunk, NPIXPERIFG, N_IFGS)
    ecl_lons_chunk = np.broadcast_to(
        ecl_lon[row_start:row_end, np.newaxis, np.newaxis], ecl_lats_chunk.shape
    ).copy()

    # Adjust latitudes to be in the range [-90, 90] (vectorized)
    mask_low = ecl_lats_chunk < -90
    ecl_lats_chunk[mask_low] = -ecl_lats_chunk[mask_low] - 180
    ecl_lons_chunk[mask_low] = 180 - ecl_lons_chunk[mask_low]
    mask_high = ecl_lats_chunk > 90
    ecl_lats_chunk[mask_high] = 180 - ecl_lats_chunk[mask_high]
    ecl_lons_chunk[mask_high] = 180 - ecl_lons_chunk[mask_high]

    t0 = utils.log_step(f"chunk {chunk_i + 1}/{n_chunks}: compute_pixel_indices", t0, args.run_name)
    pix_ecl_chunk = hp.ang2pix(nside_dust, ecl_lons_chunk, ecl_lats_chunk, lonlat=True)
    pix_ecl_firas_chunk = hp.ang2pix(g.NSIDE["firas"], ecl_lons_chunk, ecl_lats_chunk, lonlat=True)

    hit_map_counts += np.bincount(pix_ecl_firas_chunk.ravel(), minlength=npix_firas)

    t0 = utils.log_step(f"chunk {chunk_i + 1}/{n_chunks}: frankenstein_ifgs", t0, args.run_name)
    ifgs_chunk = np.empty(
        (row_end - row_start, g.NPIXPERIFG["firas"], g.N_IFGS), dtype=np.float64
    )
    for ifg_i in range(g.N_IFGS):
        ifgs_chunk[:, :, ifg_i] = ifg[pix_ecl_chunk[:, :, ifg_i], np.arange(g.NPIXPERIFG["firas"])]

    t0 = utils.log_step(f"chunk {chunk_i + 1}/{n_chunks}: sum_ifgs", t0, args.run_name)
    total_ifg_chunk = np.sum(ifgs_chunk, axis=2)

    if args.plots == "debug" and n is not None and row_start <= n < row_end:
        n_local = n - row_start
        fig, ax = plt.subplots(2, 1, figsize=(10, 8))
        ax[0].plot(ifgs_chunk[n_local], alpha=0.5)
        ax[0].set_title(f"IFGs for pixel {n}")
        ax[0].set_ylabel("Interferogram")
        ax[1].plot(total_ifg_chunk[n_local])
        ax[1].set_title(f"Total IFG for pixel {n}")
        ax[1].set_ylabel("Interferogram")
        plt.tight_layout()
        Path("../output/sims/firas/ifgs").mkdir(parents=True, exist_ok=True)
        plt.savefig(f"../output/sims/firas/ifgs/{n}_{add_on}.png")
        plt.close()
        print(f"Saved IFG {n} to ../output/sims/firas/ifgs/{n}_{add_on}.png.")

    if (args.plots in ("debug", "paper_only")) and n is not None and row_start <= n < row_end:
        n_local = n - row_start
        # Create a two-panel figure: full sky + zoomed view
        fig = plt.figure(figsize=(16, 6))

        row_pix = pix_ecl_firas_chunk[n_local]
        row_lon = ecl_lons_chunk[n_local]
        row_lat = ecl_lats_chunk[n_local]
        lon_center = float(np.mean(row_lon))
        lat_center = float(np.mean(row_lat))

        print(f"Pixels hit: {np.unique(row_pix).size} unique pixels by IFG {n}.")
        map_pix = np.bincount(row_pix.flatten(), minlength=npix_firas)
        vmax = max(1, int(map_pix.max()))

        # Left panel: full-sky mollview
        ax1 = plt.subplot(1, 2, 1)
        hp.mollview(map_pix, title="Pixels hit for one interferogram (Full Sky)", unit="Hits",
                    min=0, max=vmax, coord="E", cmap="RdYlGn", hold=True)
        hp.projplot(lon_center, lat_center, coord="E", color="blue", lonlat=True, marker="x",
                    ms=10)

        # Adjust left panel position to center it better
        ax1.set_position([0.05, 0.1, 0.4, 0.8])

        # Right panel: zoomed gnomonic view centered on the pixel
        ax2 = plt.subplot(1, 2, 2)
        hp.gnomview(map_pix, rot=(lon_center, lat_center, 0), title="Zoomed view", unit="Hits",
                    min=0, max=vmax, coord="E", cmap="RdYlGn", hold=True, xsize=800, format="%.0f")
        hp.projplot(lon_center, lat_center, coord="E", color="blue", lonlat=True, marker="x",
                    ms=10)

        # Format the axes tick labels to avoid scientific notation on the right panel
        current_ax = plt.gca()
        current_ax.ticklabel_format(style="plain", axis="both")
        current_ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.1f}"))
        current_ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.1f}"))

        Path("../output/sims/firas/pix_hits").mkdir(parents=True, exist_ok=True)
        plt.savefig(f"../output/sims/firas/pix_hits/{n}_{add_on}.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved pixel hit map for IFG {n} to ../output/sims/firas/pix_hits/{n}_{add_on}.png.")

    if args.noise:
        noise_chunk = np.random.normal(0, sigma, total_ifg_chunk.shape)
        total_ifg_chunk = total_ifg_chunk + noise_chunk

        if args.plots == "debug" and n is not None and row_start <= n < row_end:
            n_local = n - row_start
            plt.plot(total_ifg_chunk[n_local], alpha=0.5, label="Signal + Noise")
            plt.plot(np.sum(ifgs_chunk[n_local], axis=1), alpha=0.5, label="Signal")
            plt.plot(noise_chunk[n_local], alpha=0.5, label="Noise")
            plt.title(f"IFG {n} with noise")
            plt.ylabel("Interferogram")
            plt.legend()
            plt.savefig(f"../output/sims/firas/ifgs/{n}_{add_on}_with_noise.png")
            plt.close()
            print(f"Saved IFG {n} with noise to "
                  f"../output/sims/firas/ifgs/{n}_{add_on}_with_noise.png.")

    total_ifg_mm[row_start:row_end] = total_ifg_chunk
    ecl_lat_mm[row_start:row_end] = ecl_lats_chunk
    ecl_lon_mm[row_start:row_end] = ecl_lons_chunk

total_ifg_mm.flush()
ecl_lat_mm.flush()
ecl_lon_mm.flush()

if args.plots == "debug":
    t0 = utils.log_step("save_hit_map", t0, args.run_name)
    hit_map = hit_map_counts.astype(np.float64)
    mask = hit_map == 0
    hit_map[mask] = hp.UNSEEN
    if g.PNG:
        hp.mollview(hit_map, title="FIRAS Scanning Strategy Hit Map", unit="Number of hits",
                    coord=["E", "G"], format="%.0f", min=0, max=hit_map.max())

        plt.savefig(f"../output/hit_maps/scanning_strategy_firas_sim_{add_on}.png", facecolor=None,
                    bbox_inches="tight")
        plt.close()
    if g.FITS:
        hp.write_map(f"../output/hit_maps/scanning_strategy_firas_sim_{add_on}.fits", hit_map, overwrite=True,
                    dtype=np.float64)
    print("Saved hit map of the scanning strategy to ../output/hit_maps/.")

if args.noise:
    np.save(f"../output/data/firas/noise_{add_on}.npy", sigma)
print(f"Saved FIRAS IFGs to ../output/data/firas/. A total of {n_rows} IFGs were generated.")

with open(f"../output/profiling/{args.run_name}.txt", "a") as f:
    f.write(f"{(_time() - t00)/60:.2f}\n")
    f.write("=" * 50 + "\n")
    f.write(f"Total time for FIRAS simulation: {(_time() - t00)/60:.2f} min\n")
