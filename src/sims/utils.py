import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

import globals as g
import utils


def sim_dust():

    dust_map_downgraded_mjy = fits.open("../output/dust_map_downgraded.fits")
    # get map data from the fits file
    dust_map_downgraded_mjy = dust_map_downgraded_mjy[0].data

    nu0_dust = 545 * u.GHz  # Planck 2015
    A_d = 163 * u.uK
    T_d = 21 * u.K
    beta_d = 1.53

    frequencies = utils.generate_frequencies("ll", "ss", 257)

    signal = utils.dust(frequencies * u.GHz, A_d, nu0_dust, beta_d, T_d).value
    # check for invalid value encountered in divide
    signal = np.nan_to_num(signal)

    plt.plot(frequencies, signal)
    # plt.show()
    plt.savefig("../output/signal.png")
    plt.close()

    return dust_map_downgraded_mjy, frequencies, signal


def white_noise(ntod, sigma_min=0.001, sigma_max=0.1, ifg=True):
    """
    Generate white noise for the interferograms sampling the noise level from a uniform distribution.

    Parameters
    ----------
    npix : int
        How many pixels we assume each interferogram hits. For FIRAS, each interferogram will hit a maximum of 3 pixels, meaning this is the minimum number we want to use to divide the scanning strategy into multiple pixels. The higher this number, the closer to simulating a continuous scanning strategy.
    ntod : int
        Number of interferograms.
    sigma_min : float
        Standard deviation minimum value for the uniform distribution.
    sigma_max : float
        Standard deviation maximum value for the uniform distribution.
    ifg : bool
        If True, generate noise for interferograms (IFG_SIZE). If False, generate noise for spectra (SPEC_SIZE).
    Returns
    -------
    noise : array
        Array of shape (npix, ntod, IFG_SIZE) with the white noise to add to each interferogram.
    """
    sigmarand = np.random.uniform(sigma_min, sigma_max, (ntod))
    if ifg:
        size = g.IFG_SIZE
    else:
        size = g.SPEC_SIZE
    noise = np.random.normal(0, sigmarand[:, np.newaxis], (ntod, size))

    # save noise in a npz file
    np.savez("../output/white_noise.npz", noise=sigmarand)

    return noise
