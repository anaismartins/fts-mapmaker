import os
from concurrent.futures import ThreadPoolExecutor
from time import time as _time

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np

import globals as g


def ang2pix_threaded(nside, lon, lat, nworkers=None, chunk_size=5_000_000, out=None):
    """Thread-parallel hp.ang2pix (healpy's C routine releases the GIL, so this
    scales across cores instead of running single-threaded on the whole array).

    Processed in many small chunks (chunk_size elements each) fed through a bounded
    thread pool of size nworkers, rather than splitting into exactly nworkers big
    chunks. hp.ang2pix allocates several float64 temporaries per chunk, so running
    nworkers *huge* chunks at once (one per core) can multiply memory use by
    nworkers and cause swapping on large arrays; many small chunks bounds peak
    memory to roughly nworkers * chunk_size regardless of the total array size.
    """
    if nworkers is None:
        nworkers = os.cpu_count()
    if out is None:
        out = np.empty(lon.shape, dtype=np.int32)
    flat_out = out.reshape(-1)
    flat_lon = np.asarray(lon).reshape(-1)
    flat_lat = np.asarray(lat).reshape(-1)
    n_total = flat_lon.shape[0]
    n_chunks = max(1, int(np.ceil(n_total / chunk_size)))
    bounds = np.linspace(0, n_total, n_chunks + 1).astype(int)

    def work(k):
        s, e = bounds[k], bounds[k + 1]
        flat_out[s:e] = hp.ang2pix(nside, flat_lon[s:e], flat_lat[s:e], lonlat=True)

    with ThreadPoolExecutor(nworkers) as ex:
        list(ex.map(work, range(n_chunks)))
    return out


def ang2pix_cached(nside, lon, lat, cache_path, nworkers=None, chunk_size=5_000_000):
    """Like ang2pix_threaded, but caches the result to (and reloads it from) disk.

    The output is written directly to a memory-mapped .npy file instead of being
    fully materialized in RAM, to keep peak memory down for very large arrays.
    """
    if os.path.exists(cache_path):
        return np.load(cache_path, mmap_mode="r")
    out = np.lib.format.open_memmap(cache_path, mode="w+", dtype=np.int32, shape=lon.shape)
    ang2pix_threaded(nside, lon, lat, nworkers=nworkers, chunk_size=chunk_size, out=out)
    out.flush()
    return np.load(cache_path, mmap_mode="r")


def save_maps(freq, m, path, write_png=False, add_on=""):
    freq_str = f"{int(freq):04d}"
    if g.FITS:
        hp.write_map(f"{path}/{freq_str}{add_on}.fits", m, overwrite=True, dtype=np.float64)
    if g.PNG and write_png:
        hp.mollview(m, title=f"{freq_str} GHz", unit="MJy/sr", min=0, max=50, xsize=800,
                    coord=["E", "G"])
        plt.savefig(f"{path}/{freq_str}{add_on}.png")
        plt.close()

def log_step(label, t_start, run_name):
    t = _time()
    with open(f"../output/profiling/{run_name}.txt", "a") as f:
        f.write(f"{t - t_start:.2f}\n")
        f.write(f"{label:<40} | ")
    return t