"""
Conjugate gradient mapmaker that solves the equation
    A x = b
or more explicitely
    (P^T M^T N^{-1} M P) m = P^T M^T N^{-1} d.
"""

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np

import globals as g
import utils

if __name__ == "__main__":
    d = np.load("../output/ifgs.npz")["ifg"]
    d = np.roll(d, -360, axis=1)

    # sigma = np.load("../output/white_noise.npz")["noise"]

    npix = hp.nside2npix(g.NSIDE)
    # P = np.load("../input/firas_scanning_strategy.npy")

    print(
        f"Shape of d: {d.shape} and shape of P: {P.shape} and shape of sigma: {sigma.shape}"
    )

    numerator = np.zeros((npix, g.IFG_SIZE), dtype=float)
    denominator = np.zeros(npix, dtype=float)
    for xi in range(g.IFG_SIZE):
        numerator[:, xi] += (
            np.bincount(
                P,
                weights=d[:, xi] / sigma**2,
                minlength=npix,
            )
            / 3
        )
    denominator += np.bincount(P, weights=1 / sigma**2, minlength=npix) / 3

    print(
        f"Numerator and denominator calculated. Shape of denominator: {denominator.shape}"
    )
    m = np.zeros_like(numerator)
    mask = denominator == 0
    m[~mask] = numerator[~mask] / denominator[~mask][:, np.newaxis]
    m[mask] = np.nan
    # m = numerator / denominator[:, np.newaxis]
    print("Divided")
    m = np.fft.rfft(m, axis=1)

    print("Plotting maps")

    frequencies = utils.generate_frequencies("ll", "ss", 257)
    # save m as maps
    for nui in range(len(frequencies)):
        if g.FITS:
            hp.write_map(
                f"./../output/white_noise_mapmaker/{int(frequencies[nui]):04d}.fits",
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
                # norm="hist",
            )
            plt.savefig(
                f"./../output/white_noise_mapmaker/{int(frequencies[nui]):04d}.png"
            )
            plt.close()
