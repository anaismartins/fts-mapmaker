from time import time as _time

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np

import globals as g


def save_maps(freq, m):
    if g.FITS:
        hp.write_map(
            f"./../output/cg_mapmaker/{g.SIM_TYPE}/{int(freq):04d}.fits",
            m,
            overwrite=True,
            dtype=np.float64,
        )
    if g.PNG:
        hp.mollview(
            m,
            title=f"{int(freq):04d} GHz",
            unit="MJy/sr",
            min=0,
            max=50,
            xsize=2000,
            coord=["E", "G"],
        )
        plt.savefig(f"../output/cg_mapmaker/{g.SIM_TYPE}/{int(freq):04d}.png")
        plt.close()

def log_step(label, t_start, run_name):
    t = _time()
    with open(f"../output/profiling/{run_name}.txt", "a") as f:
        f.write(f"[{label}] took {t - t_start:.2f} s\n")
    return t

if __name__ == "__main__":
    # test generate_frequencies
    f_ghz = generate_frequencies(nfreq=129)