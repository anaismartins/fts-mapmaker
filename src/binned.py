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

data = np.load("../output/ifgs_modern.npz")
ifgs = data["ifg"]
pix = data["pix"]

# use only the middle pixel
pix = pix[:, pix.shape[1] // 2]
print(f"Shape of ifgs: {ifgs.shape} and shape of pix: {pix.shape}")

# plot hit map of the scanning strategy
npix = hp.nside2npix(g.NSIDE)
hit_map = np.bincount(pix, minlength=npix).astype(float)
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
    plt.savefig("../output/hit_maps/middle_pixel.png")
    plt.close()

frequencies = utils.generate_frequencies("ll", "ss", 257)

m_ifg = np.zeros((npix, g.IFG_SIZE), dtype=float)
data_density = np.zeros((npix), dtype=float)

for i in range(pix.shape[0]):
    m_ifg[pix[i]] += ifgs[i]
    data_density[pix[i]] += 1

mask = data_density == 0
m_ifg[~mask] = m_ifg[~mask] / data_density[~mask][:, np.newaxis]
m_ifg[mask] = np.nan

m = np.fft.rfft(m_ifg, axis=1)
m_abs = np.abs(m)
m_real = np.real(m)

print("Finished generating map cube, saving to disk...")

# save m as maps
for nui in range(len(frequencies)):
    if g.FITS:
        hp.write_map(
            f"../output/binned_mapmaker/{int(frequencies[nui]):04d}.fits",
            m_abs[:, nui],
            overwrite=True,
            dtype=np.float64,
        )
    if g.PNG:
        hp.mollview(
            m_abs[:, nui],
            title=f"{int(frequencies[nui]):04d} GHz",
            unit="MJy/sr",
            min=0,
            max=50,
            xsize=2000,
            coord=["E", "G"],
        )
        plt.savefig(f"../output/binned_mapmaker/{int(frequencies[nui]):04d}.png")
        plt.close()
        plt.clf()
