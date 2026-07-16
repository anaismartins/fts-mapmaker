import os
import sys

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
import globals as g
import utils

ifgs = np.load(f"../output/ifgs_{g.SIM_TYPE}.npz")
pix = np.load(f"../output/pointing_{g.SIM_TYPE}.npz")
print("Loaded IFG and pointing data from disk.")

if g.SIM_TYPE == "firas":
    ifgs = ifgs / g.N_IFGS

print(f"Shape of ifgs: {ifgs.shape} and shape of pix: {pix.shape}")

# use only the middle pixel
if g.SIM_TYPE == "fossil":
    pix = pix[:, g.NPIXPERIFG[g.SIM_TYPE] // 2]
elif g.SIM_TYPE == "firas":
    pix = pix[:, g.NPIXPERIFG[g.SIM_TYPE] // 2, g.N_IFGS // 2]
else:
    raise ValueError("g.SIM_TYPE must be 'fossil' or 'firas'")

# plot hit map of the scanning strategy
hit_map = np.bincount(pix, minlength=g.NPIX[g.SIM_TYPE]).astype(float)
mask = hit_map == 0
hit_map[mask] = hp.UNSEEN
if g.PNG:
    hp.mollview(
        hit_map,
        title="Scanning Strategy Hit Map",
        unit="Hits",
        min=0,
        max=hit_map.max(),
        xsize=2000,
        coord=["E", "G"],
    )
    plt.savefig("../output/hit_maps/binned.png")
    plt.close()

    print("Saved hit map of the scanning strategy to ../output/hit_maps/binned.png ----------------------------------------------------------------------------")

frequencies = utils.generate_frequencies(nfreq=g.SPEC_SIZE[g.SIM_TYPE], simtype=g.SIM_TYPE)

npix = g.NPIX[g.SIM_TYPE]
pix = pix.astype(np.int64, copy=False)

m_ifg = np.zeros((npix, g.IFG_SIZE[g.SIM_TYPE]), dtype=float)

# Vectorized accumulation is much faster than looping in Python.
np.add.at(m_ifg, pix, ifgs)
data_density = np.bincount(pix, minlength=npix).astype(float)

mask = data_density == 0
np.divide(m_ifg, data_density, out=m_ifg, where=~mask)
m_ifg[mask] = np.nan

m = np.fft.rfft(m_ifg, axis=1)
m_abs = np.abs(m)

print("Finished generating map cube, saving to disk...")

# save m as maps
for nui in range(len(frequencies)):
    if g.FITS:
        hp.write_map(
            f"../output/binned/{g.SIM_TYPE}/{int(frequencies[nui]):04d}.fits",
            m_abs[:, nui],
            overwrite=True,
            dtype=np.float64,
        )
    if g.PNG:
        hp.mollview(
            m_abs[:, nui],
            title=f"{int(frequencies[nui]):04d} GHz",
            unit="MJy/sr",
            # min=0,
            # max=50,
            norm='hist',
            xsize=2000,
            coord=["E", "G"],
        )
        plt.savefig(
            f"../output/binned/{g.SIM_TYPE}/{int(frequencies[nui]):04d}.png"
        )
        plt.close()
        plt.clf()

if g.PNG:
    print(f"Saved maps to ../output/binned/{g.SIM_TYPE}/ --------------------------------------------------")
