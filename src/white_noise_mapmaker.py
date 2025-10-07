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
    print(f"sigma: {sigma}")

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
    for i in range(pix.shape[0]):
        weight = 1 / sigma[i] ** 2
        numerator[pix[i]] += ifgs[i] * weight
        denominator[pix[i]] += weight
    print(
        f"Numerator and denominator calculated. Shape of numerator: {numerator.shape} and shape of denominator: {denominator.shape}"
    )

    mask = denominator == 0
    m_ifg = np.zeros((npix, g.IFG_SIZE), dtype=float)
    m_ifg[~mask] = numerator[~mask] / denominator[~mask]
    m_ifg[mask] = np.nan
    print("Divided")

    m = np.abs(np.fft.rfft(m_ifg, axis=1))
    phase = np.angle(np.fft.rfft(m_ifg, axis=1))

    print("Plotting maps")

    frequencies = utils.generate_frequencies("ll", "ss", 257)
    # save m as maps
    for nui, freq in enumerate(frequencies):
        if g.FITS:
            hp.write_map(
                f"./../output/white_noise_mapmaker/{int(freq):04d}.fits",
                m[:, nui],
                overwrite=True,
                dtype=np.float64,
            )
            hp.write_map(
                f"./../output/white_noise_mapmaker/phase_{int(freq):04d}.fits",
                phase[:, nui],
                overwrite=True,
                dtype=np.float64,
            )
        if g.PNG:
            hp.mollview(
                m[:, nui],
                title=f"{int(freq):04d} GHz",
                unit="MJy/sr",
                min=0,
                max=50,
                xsize=2000,
                coord=["E", "G"],
                # norm="hist",
            )
            plt.savefig(f"./../output/white_noise_mapmaker/{int(freq):04d}.png")
            plt.close()
            hp.mollview(
                phase[:, nui],
                title=f"Phase {int(freq):04d} GHz",
                unit="radians",
                min=-np.pi,
                max=np.pi,
                xsize=2000,
                coord=["E", "G"],
                # norm="hist",
            )
            plt.savefig(f"./../output/white_noise_mapmaker/phase_{int(freq):04d}.png")
            plt.close()
