from multiprocessing import Pool
from time import time as _time

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np

import globals as g
import paths
import spectra
import utils
import config
from config import cfg

with open(cfg.log, "w") as f:
    f.write("Profiling output for legacy mapmaker\n")
    f.write("=" * 50 + "\n")
    f.write(f"{'load ifgs':<40} | ")

t00 = _time()
t0 = _time()

config.save_record(paths.mapmaker_dir("legacy"))

ifgs = np.load(paths.data_file("ifgs"), mmap_mode="r")
t0 = utils.log_step("load_pointing", t0, cfg.log)
ecl_lon = np.load(paths.pointing_file("ecl_lon"), mmap_mode="r")
ecl_lat = np.load(paths.pointing_file("ecl_lat"), mmap_mode="r")

if cfg.sim_type == "firas":
    ifgs = ifgs / g.N_IFGS

# use only the middle pixel
t0 = utils.log_step("select_middle_pixel", t0, cfg.log)
mid_pix = g.NPIXPERIFG[cfg.sim_type] // 2
mid_ifg = g.N_IFGS // 2 if cfg.sim_type == "firas" else None

t0 = utils.log_step("load_middle_pixel", t0, cfg.log)
mid_lon_path = paths.pointing_file("ecl_lon_mid")
mid_lat_path = paths.pointing_file("ecl_lat_mid")

if mid_lon_path.exists() and mid_lat_path.exists():
    ecl_lon_mid = np.load(mid_lon_path)
    ecl_lat_mid = np.load(mid_lat_path)
else:
    # ecl_lon/ecl_lat are memory-mapped and store NPIXPERIFG (and N_IFGS) as the fastest-
    # varying axes, so indexing a single (mid_pix[, mid_ifg]) column across all rows in one
    # go forces one scattered disk read per row. Stream through the rows in large sequential
    # chunks instead (much friendlier to the filesystem), and cache the tiny result so later
    # runs skip touching the multi-GB file entirely.
    n_rows = ecl_lon.shape[0]
    ecl_lon_mid = np.empty(n_rows, dtype=ecl_lon.dtype)
    ecl_lat_mid = np.empty(n_rows, dtype=ecl_lat.dtype)
    chunk_rows = 2_000_000
    for start in range(0, n_rows, chunk_rows):
        end = min(start + chunk_rows, n_rows)
        if cfg.sim_type == "fossil":
            ecl_lon_mid[start:end] = ecl_lon[start:end, mid_pix]
            ecl_lat_mid[start:end] = ecl_lat[start:end, mid_pix]
        else:
            ecl_lon_mid[start:end] = ecl_lon[start:end, mid_pix, mid_ifg]
            ecl_lat_mid[start:end] = ecl_lat[start:end, mid_pix, mid_ifg]
    np.save(mid_lon_path, ecl_lon_mid)
    np.save(mid_lat_path, ecl_lat_mid)

if cfg.sim_type not in ("fossil", "firas"):
    raise ValueError("cfg.sim_type must be 'fossil' or 'firas'")

pix = hp.ang2pix(cfg.nside, ecl_lon_mid, ecl_lat_mid, lonlat=True)

# plot hit map of the scanning strategy
t0 = utils.log_step("create_hit_map", t0, cfg.log)
npix = cfg.npix
hit_map = np.bincount(pix, minlength=npix).astype(float)
mask = hit_map == 0
hit_map[mask] = hp.UNSEEN
if g.PNG:
    t0 = utils.log_step("plot_hit_map", t0, cfg.log)
    hp.mollview(hit_map, title="Scanning strategy hit map",
                unit="Number of hits over the full mission", min=0, max=hit_map.max(), xsize=2000,
                coord=["E", "G"])
    t0 = utils.log_step("save_hit_map", t0, cfg.log)
    hit_map_dir = paths.hit_map_dir()
    plt.savefig(hit_map_dir / "legacy.png")
    plt.savefig(hit_map_dir / "legacy.pdf")
    plt.close()

    print(f"Saved hit map of the scanning strategy to {hit_map_dir / 'legacy.png'}.")

pix = pix.astype(np.int64, copy=False)

m_ifg = np.zeros((npix, cfg.ifg_size), dtype=float)

np.add.at(m_ifg, pix, ifgs)

mask = hit_map == 0
t0 = utils.log_step("divide_by_hit_map", t0, cfg.log)
np.divide(m_ifg, hit_map[:, np.newaxis], out=m_ifg, where=~mask[:, np.newaxis])

t0 = utils.log_step("set empty to nan", t0, cfg.log)
m_ifg[mask] = np.nan

t0 = utils.log_step("rfft", t0, cfg.log)
m = np.fft.rfft(m_ifg, axis=1).real

frequencies = spectra.generate_frequencies(nfreq=cfg.spec_size, simtype=cfg.sim_type)

# save m as maps
t0 = utils.log_step("save_maps", t0, cfg.log)
out_dir = paths.mapmaker_dir("legacy")
args_list = [(frequencies[nui], m[:, nui], out_dir, cfg.plots == "debug")
             for nui in range(len(frequencies))]

# hp.mollview + savefig dominates the per-frequency cost, so fan out across processes
with Pool(processes=cfg.nworkers) as pool:
    list(pool.starmap(utils.save_maps, args_list))

print(f"Saved maps to {out_dir}.")
    
with open(cfg.log, "a") as f:
    f.write(f"{(_time() - t0):0.2f}\n")
    f.write("=" * 50 + "\n")
    f.write(f"Total time for legacy mapmaker: {(_time() - t00)/60:.2f} min\n")