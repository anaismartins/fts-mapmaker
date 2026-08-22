import multiprocessing

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np

import globals as g
import spectra

# multiply saved cg fits files by 16
frequencies = spectra.generate_frequencies(simtype="firas", nfreq=g.SPEC_SIZE["firas"])
# print(frequencies)

def multiply_cg_files(nu_i):
    m = hp.read_map(f"../output/cg/firas/{int(frequencies[nu_i]):04d}.fits")
    hp.write_map(f"../output/cg/firas/{int(frequencies[nu_i]):04d}.fits", m / 16, overwrite=True,
                 dtype=np.float64)

def plot_cg_maps(nu_i):
    m = hp.read_map(f"../output/cg/firas/{int(frequencies[nu_i]):04d}.fits")
    hp.mollview(m, title=f"CG Map at {int(frequencies[nu_i])} GHz", min=0, max=50, cbar=True,
                coord=["E", "G"], cmap="viridis")
    plt.savefig(f"../output/cg/firas/{int(frequencies[nu_i]):04d}.png")
    plt.close()


with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
    pool.map(multiply_cg_files, range(len(frequencies)))

with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
    pool.map(plot_cg_maps, range(len(frequencies)))