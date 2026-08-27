from time import time as _time

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np

import globals as g


def save_maps(freq, m, path, write_png=False, add_on=""):
    freq_str = f"{int(freq):04d}"
    if g.FITS:
        hp.write_map(f"{path}/{freq_str}_{add_on}.fits", m, overwrite=True, dtype=np.float64)
    if g.PNG and write_png:
        hp.mollview(m, title=f"{freq_str} GHz", unit="MJy/sr", min=0, max=50, xsize=800,
                    coord=["E", "G"])
        plt.savefig(f"{path}/{freq_str}_{add_on}.png")
        plt.close()

def _save_one_map(args):
    freq, dust_map_i, out_dir, add_on = args
    # adjust if utils.save_maps signature is different
    save_maps(freq, dust_map_i, out_dir, write_png=True, add_on=add_on)

def log_step(label, t_start, run_name):
    t = _time()
    with open(f"../output/profiling/{run_name}.txt", "a") as f:
        f.write(f"{t - t_start:.2f}\n")
        f.write(f"{label:<40} | ")
    return t