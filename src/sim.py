import os
import sys
import time

import astropy.units as u
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

from globals import IFG_SIZE, SPEC_SIZE
from utils import dust

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
import globals as g
import utils


def sim_dust():

    dust_map_downgraded_mjy = fits.open("../output/dust_map_downgraded.fits")
    # get map data from the fits file
    dust_map_downgraded_mjy = dust_map_downgraded_mjy[0].data

    nu0_dust = 545 * u.GHz # Planck 2015
    A_d = 163 * u.uK
    T_d = 21 * u.K
    beta_d = 1.53

    frequencies = utils.generate_frequencies("ll", "ss", 257)

    signal = dust(frequencies * u.GHz, A_d, nu0_dust, beta_d, T_d).value
    # check for invalid value encountered in divide
    signal = np.nan_to_num(signal)
    
    plt.plot(frequencies, signal)
    # plt.show()
    plt.savefig("../output/signal.png")
    plt.close()

    return dust_map_downgraded_mjy, frequencies, signal

def white_noise(ntod, sigma_min=0.001, sigma_max=0.1):
    sigmarand = np.random.uniform(sigma_min, sigma_max, ntod)
    noise = np.random.normal(0, sigmarand[:, np.newaxis], (ntod, IFG_SIZE))

    # save noise in a npz file
    np.savez("../output/white_noise.npz", noise=sigmarand)

    return noise

if __name__ == "__main__":
    dust_map_downgraded_mjy, frequencies, sed = sim_dust()
    sed = np.nan_to_num(sed)

    spec = dust_map_downgraded_mjy[:, np.newaxis] * sed[np.newaxis, :]

    print("Calculating and plotting IFGs")

    # time ifg making
    time_start = time.time()

    ifg = np.fft.irfft(spec, axis=1)

    # add phase to ifg
    ifg = np.roll(ifg, 360, axis=1)
    # turn ifg into real signal
    ifg = ifg.real

    # introduce scanning strategy
    # pix_gal = np.load("../input/firas_scanning_strategy.npy").astype(int)
    pix_ecl = np.load("../input/firas_scanning_strategy.npy").astype(int)
    print(f"Shape of pix_ecl: {pix_ecl.shape}")

    ifg_scanning = np.zeros((len(pix_ecl), IFG_SIZE))
    spec_scanning = np.zeros((len(pix_ecl), SPEC_SIZE))
    print("Calculating IFGs for scanning strategy")
    for j in range(3):
        for i, pix in enumerate(pix_ecl[:, j]):
            ifg_scanning[i] += ifg[pix]/3
            spec_scanning[i] += spec[pix]/3

    # add noise to ifg
    ifg_scanning = ifg_scanning + white_noise(ifg_scanning.shape[0])

    # bin spec_scanning into spec maps
    spec_map = np.zeros((g.NPIX, g.SPEC_SIZE))
    dd = np.zeros(g.NPIX)
    for i in range(pix_ecl.shape[0]):
        for j in range(pix_ecl.shape[1]):
            spec_map[pix_ecl[i, j]] += spec_scanning[i] / 3
            dd[pix_ecl[i, j]] += 1 / 3

    mask = dd == 0
    spec_map[~mask] = spec_map[~mask] / dd[~mask][:, np.newaxis]
    spec_map[mask] = np.nan

    for i, freq in enumerate(frequencies):
        if g.PNG:
            hp.mollview(spec_map[:, i], title=f"{int(freq):04d} GHz", unit="MJy/sr", min=0, max=200, xsize=2000, coord=['E', 'G'])
            plt.savefig(f"../output/sim_maps/{int(freq):04d}.png")
            plt.close()
        if g.FITS:
            hp.write_map(f"../output/sim_maps/{int(freq):04d}.fits", spec_map[:, i], overwrite=True)

    # plt.plot(ifg_scanning[np.random.randint(0, ifg_scanning.shape[0]), :], label="IFG 1")
    # plt.plot(ifg_scanning[np.random.randint(0, ifg_scanning.shape[0]), :], label="IFG 2")
    # plt.plot(ifg_scanning[np.random.randint(0, ifg_scanning.shape[0]), :], label="IFG 3")
    # plt.xlabel("Spacing")
    # plt.ylabel("Signal (mJy)")
    # plt.title("Interferogram")
    # plt.legend()
    # plt.show()
    # plt.savefig("../output/interferogram.png")
    # plt.close()

    print("Saving IFGs to file")

    # save ifg products in a npz file
    np.savez("../output/ifgs.npz", ifg=ifg_scanning)

    time_end = time.time()
    print(f"Time elapsed for IFGs: {(time_end - time_start)/60} minutes")
