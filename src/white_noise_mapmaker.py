"""
Maximum likelihood mapmaker that solves the equation
    (P^T M^T N^{-1} M P) m = P^T M^T N^{-1} d,
assuming there is only white noise i.e. N is diagonal, which means the equation reduces to
    m = sum (d / sigma ^2) / sum (1 / sigma^2).
"""

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np

import globals as g
import utils
from argparser import args

if __name__ == "__main__":
    data = np.load(f"../output/ifgs_{args.sim_type}.npz")
    ifgs = data["ifg"]
    pix = data["pix"]
    sigma = data["sigma"]
    print(f"sigma: {sigma}")

    # # plot ifgs
    # plt.imshow(ifgs, aspect="auto")
    # plt.colorbar(label="Interferogram")
    # plt.xlabel("IFG point")
    # plt.ylabel("IFG number")
    # plt.title("Input IFGs")
    # plt.savefig("../output/ifgs.png")
    # plt.close()

    print(
        f"Shape of ifgs: {ifgs.shape} and shape of P: {pix.shape} and shape of sigma: {sigma.shape}"
    )

    if args.sim_type == "firas":
        ifgs = ifgs / g.N_IFGS

    ifgs = np.roll(ifgs, -360, axis=1)

    # how many unique pixels are there?
    unique_pixels = np.unique(pix)
    print(f"Number of unique pixels in P: {len(unique_pixels)} out of {g.NPIX}")

    numerator = np.zeros((g.NPIX, g.IFG_SIZE[args.sim_type]), dtype=float)
    denominator = np.zeros((g.NPIX, g.IFG_SIZE[args.sim_type]), dtype=float)
    # Vectorized accumulation: loop over IFG sample index (usually much smaller
    # than the number of IFGs) and use np.bincount to accumulate values per pixel.
    # This avoids the expensive Python-level loop over all IFGs and is much faster.
    weights = 1.0 / (sigma**2)
    if args.sim_type == "fossil":
        for x_i in range(g.IFG_SIZE[args.sim_type]):
            pix_s = pix[:, x_i]
            vals = ifgs[:, x_i] * weights
            # bincount returns length npix; fill the column x_i for numerator/denominator
            numerator[:, x_i] = np.bincount(pix_s, weights=vals, minlength=g.NPIX)
            denominator[:, x_i] = np.bincount(pix_s, weights=weights, minlength=g.NPIX)
    elif args.sim_type == "firas":
        for ifg_i in range(g.N_IFGS):
            # for i in range(pix.shape[1]):
            #     numerator[pix[ifg_i, i]] += ifgs[i] * weights[i]
            #     denominator[pix[ifg_i, i]] += weights[i]
            for x_i in range(g.IFG_SIZE[args.sim_type]):
                pix_s = pix[ifg_i, :, x_i]
                vals = ifgs[:, x_i] * weights
                # bincount returns length npix; fill the column x_i for numerator/denominator
                numerator[:, x_i] += np.bincount(pix_s, weights=vals, minlength=g.NPIX)
                denominator[:, x_i] += np.bincount(
                    pix_s, weights=weights, minlength=g.NPIX
                )
    else:
        raise ValueError(f"Unknown sim_type: {args.sim_type}")
    
    print(
        f"Numerator and denominator calculated. Shape of numerator: {numerator.shape} and shape of denominator: {denominator.shape}"
    )

    mask = denominator == 0

    hp.mollview(
        numerator[:, 100],
        title="Numerator at frequency index 100",
        coord=["E", "G"],
        min=0,
        max=200,
    )
    plt.show()
    hp.mollview(
        denominator[:, 100],
        title="Denominator at frequency index 100",
        coord=["E", "G"],
        min=0,
        max=200,
    )
    plt.savefig("../output/white_noise_mapmaker/fossil/denominator.png")
    # check if denominator is all zeros
    print(f"Number of zeros in denominator: {np.sum(denominator == 0)}")

    m_ifg = np.zeros((g.NPIX, g.IFG_SIZE[args.sim_type]), dtype=float)
    m_ifg[~mask] = numerator[~mask] / denominator[~mask]
    m_ifg[mask] = np.nan
    print("Divided")

    # check if m_ifg has any nans
    print(f"Number of NaNs in m_ifg: {np.isnan(m_ifg).sum()}")

    # save the ifg maps
    for i in range(g.IFG_SIZE[args.sim_type]):
        path = f"./../output/white_noise_mapmaker/{args.sim_type}/"
        if g.FITS:
            hp.write_map(f"{path}ifg_maps/ifg_{i:04d}.fits", m_ifg[:, i], overwrite=True,
                         dtype=np.float64)
        if g.PNG:
            hp.mollview(m_ifg[:, i], title=f"IFG map at point index {i:04d}", unit="Amplitude",
                        coord=["E", "G"])
            plt.savefig(f"{path}ifg_maps/ifg_{i:04d}.png")
            plt.close()

    m = np.abs(np.fft.rfft(m_ifg, axis=1))
    phase = np.angle(np.fft.rfft(m_ifg, axis=1))

    print("Plotting maps")

    if args.sim_type == "fossil":
        nfreq = 129
    elif args.sim_type == "firas":
        nfreq = 257
    frequencies = utils.generate_frequencies(nfreq=nfreq)
    # save m as maps
    for nui, freq in enumerate(frequencies):
        if g.FITS:
            hp.write_map(f"{path}maps/{int(freq):04d}.fits", m[:, nui], overwrite=True, dtype=np.float64)
            hp.write_map(f"{path}phase_maps/{int(freq):04d}.fits", phase[:, nui], overwrite=True,
                         dtype=np.float64)
        if g.PNG:
            hp.mollview(m[:, nui], title=f"{int(freq):04d} GHz", unit="MJy/sr", min=0, max=50,
                        xsize=2000, coord=["E", "G"])
            plt.savefig(f"{path}maps/{int(freq):04d}.png")
            plt.close()
            hp.mollview(phase[:, nui], title=f"Phase {int(freq):04d} GHz", unit="radians",
                        min=-np.pi, max=np.pi, xsize=2000, coord=["E", "G"])
            plt.savefig(f"{path}phase_maps/{int(freq):04d}.png")
            plt.close()
