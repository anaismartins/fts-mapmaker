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
    data = np.load("../output/ifgs_modern.npz")
    ifgs = data["ifg"]
    pix = data["pix"]
    sigma = data["sigma"]

    npixperifg = pix.shape[1]

    # plot ifgs
    plt.imshow(ifgs, aspect="auto")
    plt.colorbar(label="Interferogram")
    plt.xlabel("IFG point")
    plt.ylabel("IFG number")
    plt.title("Input IFGs")
    plt.savefig("../output/ifgs.png")
    plt.close()

    ifgs = np.roll(ifgs, -360, axis=1)

    print(
        f"Shape of ifgs: {ifgs.shape} and shape of P: {pix.shape} and shape of sigma: {sigma.shape}"
    )

    npix = hp.nside2npix(g.NSIDE)

    # hit map
    hit_map = np.bincount(pix.flatten(), minlength=npix).astype(float)
    mask = hit_map == 0
    hit_map[mask] = np.nan
    if g.PNG:
        hp.mollview(
            hit_map / npixperifg,
            title="Scanning Strategy Hit Map",
            unit="Hits",
            min=0,
            max=np.nanmax(hit_map / npixperifg),
            xsize=2000,
            coord=["E", "G"],
        )
        plt.savefig("../output/hit_maps/scanning_strategy.png")
        plt.close()

    # how wmany unique pixels are there?
    unique_pixels = np.unique(pix)
    print(f"Number of unique pixels in P: {len(unique_pixels)} out of {npix}")

    numerator = np.zeros((npix, g.IFG_SIZE), dtype=float)
    denominator = np.zeros((npix, g.IFG_SIZE), dtype=float)

    # # for xi in range(g.IFG_SIZE):
    # # numerator = np.bincount(
    # #     pix,
    # #     weights=ifgs,  # / sigma**2,
    # #     minlength=npix,
    # # )
    # numerator = bincount2d(pix, bins=npix) @ ifgs  # / sigma**2,
    # # denominator = np.bincount(pix, minlength=npix)  # / sigma**2,
    # denominator = bincount2d(pix, bins=npix)  # @ (1 / sigma**2)
    # Efficient accumulation using np.add.at

    # repeat pix to match ifgs shape
    

    weight = 1 / sigma[:, None] ** 2
    np.add.at(numerator, (pix, np.arange(g.IFG_SIZE)), ifgs * weight)
    np.add.at(denominator, (pix, np.arange(g.IFG_SIZE)), weight)

    # numerator = np.histogram2d(pix, weights=ifgs, bins=npix)
    # ifgs = ifgs.flatten()
    # pix = pix.flatten()

    # for i in range(g.IFG_SIZE):
    #     pix[i * len(pix) // g.IFG_SIZE : (i + 1) * len(pix) // g.IFG_SIZE] = (
    #         pix[i * len(pix) // g.IFG_SIZE : (i + 1) * len(pix) // g.IFG_SIZE] * i
    #     )
    # sigma = np.repeat(sigma, g.IFG_SIZE)
    # numerator = np.bincount(pix, weights=ifgs / sigma**2, minlength=npix * g.IFG_SIZE)
    # denominator = np.bincount(pix, weights=1 / sigma**2, minlength=npix * g.IFG_SIZE)

    print(
        f"Numerator and denominator calculated. Shape of numerator: {numerator.shape} and shape of denominator: {denominator.shape}"
    )
    # m = np.zeros_like(numerator)
    # mask = denominator == 0
    # m[~mask] = numerator[~mask] / denominator[~mask]
    # m[mask] = np.nan
    m = numerator / denominator

    # reshape m to (npix, IFG_SIZE)
    # m = m.reshape((npix, g.IFG_SIZE))
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
