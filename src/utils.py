from time import time as _time

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np

import globals as g


def save_maps(freq, m, path):
    if g.FITS:
        hp.write_map(f"{path}/{int(freq):04d}.fits", m, overwrite=True, dtype=np.float64)
    if g.PNG:
        hp.mollview(m, title=f"{int(freq):04d} GHz", unit="MJy/sr", min=0, max=50, xsize=2000,
                    coord=["E", "G"])
        plt.savefig(f"{path}/{int(freq):04d}.png")
        plt.close()

def log_step(label, t_start, run_name):
    t = _time()
    with open(f"../output/profiling/{run_name}.txt", "a") as f:
        f.write(f"{label:<30} | {t - t_start:.2f}\n")
    return t