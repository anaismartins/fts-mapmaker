import astropy.constants as const
import astropy.units as u
import numpy as np

import globals as g


def white_noise(ntod, simtype):
    """
    Generate white noise for the interferograms sampling the noise level from a uniform
    distribution.

    Parameters
    ----------
    ntod : int
        Number of interferograms.
    simtype : str
        Type of simulation, e.g. "fossil" or "firas".
    Returns
    -------
    noise : array
        Array of shape (npix, ntod, IFG_SIZE) with the white noise to add to each interferogram.
    """
    if simtype == "fossil":
        sigma_full_mission = 1 * u.Jy / u.sr
        sigma_jy = sigma_full_mission * np.sqrt(ntod)
        sigma_Mjy = (sigma_jy.to(u.MJy / u.sr)).value
    elif simtype == "firas":
        # from calibration paper
        sigma_uerg = 0.1 * np.sqrt(93) * u.uerg / u.s / u.cm**2 / u.sr * u.cm / const.c
        sigma_Mjy = (sigma_uerg.to(u.MJy / u.sr)).value

    noise = np.random.normal(0, sigma_Mjy, (ntod, g.IFG_SIZE[simtype]))
    print(f"Generated white noise with sigma = {sigma_Mjy:.3f} MJy/sr.")
    return noise, sigma_Mjy