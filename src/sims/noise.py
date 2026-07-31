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
    sigma = None
    print(f"DEBUG: Number of TODs: {ntod}")
    
    if simtype == "fossil":
        sigma = np.full(g.IFG_SIZE[simtype], 1e-6 * np.sqrt(ntod))
    elif simtype == "firas":
        # from calibration paper
        sigma_uerg = 0.1 * np.sqrt(93) * u.uerg / u.s / u.cm**2 / u.sr * u.cm / const.c
        sigma_Mjy = (sigma_uerg.to(u.MJy / u.sr)).value
        sigma = np.full(g.IFG_SIZE[simtype], sigma_Mjy)
    if sigma is None:
        raise ValueError("Could not derive noise sigma; check simtype/ifg configuration.")

    noise = np.random.normal(0, sigma[np.newaxis, :], (ntod, g.IFG_SIZE[simtype]))
    return noise, sigma