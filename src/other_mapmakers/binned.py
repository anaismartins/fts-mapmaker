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

d = np.load("../../output/ifgs.npz")["ifg"]
pix = np.load("../../input/firas_scanning_strategy.npy")

# plot hit map of the scanning strategy
hit_map = np.bincount(pix[:, 1], minlength=hp.nside2npix(g.NSIDE)).astype(float)
mask = hit_map == 0
hit_map[mask] = hp.UNSEEN
hp.mollview(
    hit_map,
    title="Scanning Strategy Hit Map",
    unit="Hits",
    min=0,
    max=hit_map.max(),
    xsize=2000,
)
plt.savefig("../../output/scanning_strategy_hit_map.png")
plt.close()

npix = hp.nside2npix(g.NSIDE)
frequencies = utils.generate_frequencies("ll", "ss", 257)

m = np.zeros((npix, 257), dtype=complex)
data_density = np.zeros(npix, dtype=float)

for i in range(pix.shape[0]):
    for j in range(pix.shape[1]):
        m[pix[i, j]] += np.abs(np.fft.rfft(d[i])) / 3
        data_density[pix[i, j]] += 1 / 3

mask = data_density == 0
m[~mask] = m[~mask] / data_density[~mask][:, np.newaxis]
m[mask] = np.nan

print("Finished generating map cube, saving to disk...")

# save m as maps
for nui in range(len(frequencies)):
    if g.FITS:
        hp.write_map(
            f"../../output/binned_mapmaker/{int(frequencies[nui]):04d}.fits",
            np.abs(m[:, nui]),
            overwrite=True,
        )
    if g.PNG:
        hp.mollview(
            np.abs(m[:, nui]),
            title=f"{int(frequencies[nui]):04d} GHz",
            unit="MJy/sr",
            min=0,
            max=200,
            xsize=2000,
            coord=["E", "G"],
        )
        plt.savefig(f"../../output/binned_mapmaker/{int(frequencies[nui]):04d}.png")
        plt.close()
        plt.clf()
