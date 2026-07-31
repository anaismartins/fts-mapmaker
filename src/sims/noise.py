import numpy as np
from astropy.io import fits

import globals as g


def white_noise(ntod, simtype, args, ifg=True, signal=None):
    """
    Generate white noise for the interferograms sampling the noise level from a uniform
    distribution.

    Parameters
    ----------
    ntod : int
        Number of interferograms.
    simtype : str
        Type of simulation, e.g. "fossil" or "firas".
    ifg : bool
        If True, generate noise for interferograms (IFG_SIZE). If False, generate noise for spectra
        (SPEC_SIZE).
    signal : array, optional
        The signal array to determine the noise level.
    Returns
    -------
    noise : array
        Array of shape (npix, ntod, IFG_SIZE) with the white noise to add to each interferogram.
    """
    sigma = None
    print(f"DEBUG: Number of TODs: {ntod}")
    
    if not ifg:
        if simtype == "fossil":
            sigma = np.full(g.IFG_SIZE[simtype], 1e-6 * np.sqrt(ntod))
        elif simtype == "firas":
            firas_noise = fits.open("sims/FIRAS_CALIBRATION_ERRORS_LHSS.FITS")
            print(firas_noise.info()) # TODO: check this and plot against calibration paper, figure 9
            raise NotImplementedError("FIRAS noise model is not implemented yet.")

    if sigma is None:
        raise ValueError("Could not derive noise sigma; check simtype/ifg configuration.")

    noise = np.random.normal(0, sigma[np.newaxis, :], (ntod, g.IFG_SIZE[simtype]))
    return noise, sigma