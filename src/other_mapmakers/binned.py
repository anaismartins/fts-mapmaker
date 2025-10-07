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

# d = np.load("../../output/ifgs.npz")["ifg"]
data = np.load("../output/ifgs_modern.npz")
ifgs = data["ifg"]
pix = data["pix"]
print(f"Shape of ifgs: {ifgs.shape} and shape of pix: {pix.shape}")

# plot hit map of the scanning strategy
npix = hp.nside2npix(g.NSIDE)
hit_map = np.bincount(pix.flatten(), minlength=npix).astype(float)
mask = hit_map == 0
hit_map[mask] = hp.UNSEEN
if g.PNG:
    hp.mollview(
        hit_map / 512,
        title="Scanning Strategy Hit Map",
        unit="Hits",
        min=0,
        max=(hit_map / 512).max(),
        xsize=2000,
        coord=["E", "G"],
    )
    plt.savefig("../output/hit_maps/scanning_strategy.png")
    plt.close()

frequencies = utils.generate_frequencies("ll", "ss", 257)

m_ifg = np.zeros((npix, g.IFG_SIZE), dtype=float)
data_density = np.zeros((npix), dtype=float)

for i in range(pix.shape[0]):
    m_ifg[pix[i]] += ifgs[i]
    # spec = np.abs(np.fft.rfft(ifgs[i]))
    # print(f"pix[i].shape: {pix[i].shape} and spec shape: {spec.shape}")
    # m[pix[i]] += spec
    data_density[pix[i]] += 1
    # print(f"pix hit: {pix[i]}")
    # print(f"data_density[pix[i]] shape: {data_density[pix[i]].shape}")
    # print(f"data_density[pix[i]]: {data_density[pix[i]]}")

mask = data_density == 0
m_ifg[~mask] = m_ifg[~mask] / data_density[~mask][:, np.newaxis]
m_ifg[mask] = np.nan

m = np.abs(np.fft.rfft(m_ifg, axis=1))

print("Finished generating map cube, saving to disk...")

# save m as maps
for nui in range(len(frequencies)):
    if g.FITS:
        hp.write_map(
            f"../output/binned_mapmaker/{int(frequencies[nui]):04d}.fits",
            np.abs(m[:, nui]),
            overwrite=True,
            dtype=np.float64,
        )
    if g.PNG:
        hp.mollview(
            np.abs(m[:, nui]),
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
