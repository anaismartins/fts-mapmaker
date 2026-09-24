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
        plt.savefig(f"{path}/{freq_str}{add_on}.png")
        plt.close()

def log_step(label, t_start, run_name):
    t = _time()
    with open(f"../output/profiling/{run_name}.txt", "a") as f:
        f.write(f"{t - t_start:.2f}\n")
        f.write(f"{label:<40} | ")
    return t