"""
Conjugate gradient mapmaker that solves the equation
    A x = b
or more explicitely
    (P^T M^T N^{-1} M P) m = P^T M^T N^{-1} d.
"""

import os
import sys

import globals as g
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import utils

if __name__ == "__main__":
    d = np.load("../output/ifgs.npz")["ifg"]
    d = np.roll(d, -360, axis=1)

    ntod = d.shape[0]
    sigma = np.load("../output/white_noise.npz")["noise"]

    npix = hp.nside2npix(g.NSIDE)
    P = np.load("../input/firas_scanning_strategy.npy")

    numerator = np.zeros((npix, d.shape[1]), dtype=float)
    denominator = np.zeros(npix, dtype=float)
    for pixi in range(3):
        for xi in range(d.shape[1]):
            numerator[:, xi] += np.bincount(P[:, pixi], weights=d[:, xi]/sigma**2, minlength=npix) / 3
        denominator += np.bincount(P[:, pixi], weights=1/sigma**2, minlength=npix) / 3
    m = numerator / denominator[:, np.newaxis]
    m = np.fft.rfft(m, axis=1)

    frequencies = utils.generate_frequencies("ll", "ss", 257)
    # save m as maps
    for nui in range(len(frequencies)):
        if g.FITS:
            hp.write_map(f"./../output/white_noise_mapmaker/{int(frequencies[nui]):04d}.fits", np.abs(m[:, nui]), overwrite=True)
        if g.PNG:
            hp.mollview(
                np.abs(m[:, nui]),
                title=f"{int(frequencies[nui]):04d} GHz",
                unit="MJy/sr",
                min=0,
                max=200,
                xsize=2000,
                coord=["E", "G"],
                # norm='hist',
            )
            plt.savefig(f"./../output/white_noise_mapmaker/{int(frequencies[nui]):04d}.png")
            plt.close()
            plt.clf()